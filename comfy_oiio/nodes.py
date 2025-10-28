import re
import gc
import heapq
import itertools
import json
import glob
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Any, List

import folder_paths
import numpy as np
import OpenImageIO as oiio
import torch
from OpenImageIO import ImageSpec
import clique

log = getLogger("comfy-oiio")

DEFAULT_INPUT_TRANSFORM = "scene_linear"
DEFAULT_OUTPUT_TRANSFORM = "sRGB"

# TODO: Add support for sublayers / arbitrary channels
# TODO: Support Display/View transforms? (probably not easy to adapt to comfy)
# TODO: Better error handling


class OIIOUtils:
    @staticmethod
    def comfy_to_oiio(tensor: torch.Tensor) -> tuple[oiio.ImageBuf, oiio.ROI]:
        """Convert a Comfy tensor to OIIO ImageBuf and ROI

        Args:
            tensor: [B,H,W,C] torch tensor
        Returns:
            buf: OIIO ImageBuf
            roi: OIIO ROI
        """
        pixels = tensor.float()

        if pixels.dim() == 4:
            pixels = pixels.squeeze(0)

        pixels = pixels.cpu().numpy()
        pixels = np.ascontiguousarray(pixels)

        spec = oiio.ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], oiio.FLOAT)
        roi = oiio.ROI(0, pixels.shape[1], 0, pixels.shape[0], 0, 1, 0, pixels.shape[2])

        buf = oiio.ImageBuf(spec)
        buf.set_pixels(roi, pixels)

        return buf, roi

    @staticmethod
    def oiio_to_comfy(buf: oiio.ImageBuf, roi: oiio.ROI, add_batch: bool = True) -> torch.Tensor:
        """Convert OIIO ImageBuf back to Comfy tensor

        Args:
            buf: OIIO ImageBuf
            roi: OIIO ROI
            add_batch: Whether to add batch dimension
        Returns:
            torch tensor in Comfy format [B,H,W,C] or [H,W,C]
        """
        pixels = torch.from_numpy(buf.get_pixels(roi=roi))

        if add_batch:
            pixels = pixels.unsqueeze(0)

        return pixels

    @staticmethod
    def convert_colorspace(
        buf: oiio.ImageBuf, from_colorspace: str, to_colorspace: str, roi: oiio.ROI | None = None
    ) -> oiio.ImageBuf:
        """Convert ImageBuf between colorspaces

        Args:
            buf: Input ImageBuf
            from_colorspace: Source colorspace
            to_colorspace: Target colorspace
            roi: Optional ROI (will use buf's ROI if None)

        Returns
        -------
            Converted ImageBuf
        Raises:
            RuntimeError if conversion fails
        """
        if from_colorspace == "auto" or to_colorspace == "auto" or from_colorspace == to_colorspace:
            return buf

        out_buf = oiio.ImageBuf(buf.spec())

        if not oiio.ImageBufAlgo.colorconvert(out_buf, buf, from_colorspace, to_colorspace):
            raise RuntimeError(f"Color conversion failed: {out_buf.geterror()}")

        return out_buf

    @staticmethod
    def get_image_stats(tensor: torch.Tensor) -> str:
        """Get debug string with image statistics

        Args:
            tensor: Input tensor
        Returns:
            Formatted string with stats
        """
        return (
            f"Shape: {tensor.shape}, "
            f"dtype: {tensor.dtype}, "
            f"range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}], "
            f"mean: {tensor.mean().item():.6f}"
        )


def retrieve_sequence(filepath:str) -> List[str]:
    # final result: dir/file.%04d.exr [100-200]

    # if missing frame range, glob into the folder
    if " [" not in filepath:
        filepath = re.sub(r"#+", "*", filepath)
        filepath = re.sub(r"%0\d+d", "*", filepath)

        sequences, files = clique.assemble(
            glob.glob(filepath),
            patterns=[clique.PATTERNS["frames"]],
            assume_padded_when_ambiguous=True,
        )
        # find only one file
        if files:
            return [files[0]]
        if sequences:
            return list(sequences[0])

        raise ValueError(f"No readable images found in {filepath}")

    # file.####.exr [1-2] => file.%04d.exr [1-2]
    elif "#" in filepath:
        padding = filepath.count("#")
        filepath = filepath.replace("#"*padding, f"%0{padding}d")
    
    return list(clique.parse(filepath))


class OIIO_LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        colorspaces = OIIO_ColorspaceConvert.get_colorspaces()
        return {
            "required": {
                "filepath": ("STRING", {"default": ""}),
                "precision": (["auto", "uint8", "half", "float"], {"default": "auto"}),
                "input_transform": (
                    colorspaces,
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "xres", "yres")
    FUNCTION = "read"
    CATEGORY = "oiio"

    def read(self, filepath="", precision="auto", input_transform="auto"):
        path = Path(filepath)

        if path.is_dir() or "%" in filepath or "#" in filepath:
            content = path.iterdir()
            if not path.is_dir():
                content = retrieve_sequence(filepath)
                log.debug(f"Load {content.__len__()} images from sequence")

            pixels_list = []
            masks_list = []
            xres = yres = None
            for f in sorted(content):
                try:
                    result = self.read_single(f, precision, input_transform)
                    if result is None:
                        continue

                    # use the first match as reference for dimensions
                    if xres is None:
                        xres = result[2]
                        yres = result[3]
                    elif (result[2], result[3]) != (xres, yres):
                        log.warning(f"Warning: Skipping {f}, dimensions don't match")
                        continue

                    pixels_list.append(result[0])
                    masks_list.append(result[1])

                except Exception as e:
                    log.error(f"Warning: Couldn't read {f}: {e}")
                    continue

            if not pixels_list:
                raise ValueError(f"No readable images found in {filepath}")

            # stack
            pixels_tensor = torch.cat(pixels_list, dim=0)
            mask_tensor = torch.cat(masks_list, dim=0)

            return (pixels_tensor, mask_tensor, xres, yres)

        else:
            return self.read_single(path, precision, input_transform)

    def read_single(self, filepath, precision, input_transform):
        """Read a single image file."""
        img = oiio.ImageInput.open(str(filepath))
        if img:
            spec = img.spec()
            xres = spec.width
            yres = spec.height
            nchannels = spec.nchannels

            if precision == "auto":
                read_type = spec.format
            else:
                type_map = {"uint8": oiio.UINT8, "half": oiio.HALF, "float": oiio.FLOAT}
                read_type = type_map.get(precision, spec.format)

            pixels = img.read_image(read_type)
            img.close()

            if input_transform != "auto":
                in_spec = oiio.ImageSpec(xres, yres, nchannels, oiio.FLOAT)
                roi = oiio.ROI(0, xres, 0, yres, 0, 1, 0, nchannels)

                in_buf = oiio.ImageBuf(in_spec)
                in_buf.set_pixels(roi, pixels)

                out_buf = oiio.ImageBuf(in_spec)

                if not oiio.ImageBufAlgo.colorconvert(
                    out_buf,
                    in_buf,
                    spec.get_string_attribute("oiio:ColorSpace", "sRGB"),
                    input_transform,
                ):
                    raise RuntimeError(f"Color conversion failed: {out_buf.geterror()}")

                pixels = out_buf.get_pixels()

            if precision not in ("float", "auto"):
                pixels = pixels.astype(precision)

            rgb = pixels[:, :, :3] if nchannels > 3 else pixels
            pixels_tensor = torch.from_numpy(rgb)

            if pixels_tensor.dtype == torch.uint8:
                pixels_tensor = pixels_tensor.float() / 255.0

            if pixels_tensor.dim() == 3:
                pixels_tensor = pixels_tensor.unsqueeze(0)

            if nchannels > 3:
                mask_tensor = torch.from_numpy(pixels[:, :, 3]).float()
                if mask_tensor.dtype == torch.uint8:
                    mask_tensor = mask_tensor / 255.0
            else:
                mask_tensor = torch.ones((yres, xres), dtype=torch.float32)

            mask_tensor = mask_tensor.unsqueeze(0)

            return (pixels_tensor, mask_tensor, xres, yres)
        else:
            log.error(f"Warning: Couldn't open {filepath}: {oiio.geterror()}")

        return None


class OIIO_ColorspaceConvert:
    @classmethod
    def get_colorspaces(cls):
        common = [
            "scene_linear",
            "lin_srgb",
            "lin_rec709",
            "Rec709",
            "sRGB",
            "linear",
            "ACEScg",
            "AdobeRGB",
            "KodakLog",
            "g22_rec709",
            "g18_rec709",
            "Gamma2.4",
        ]
        colorspaces = []
        try:
            config = oiio.ColorConfig()
            colorspaces = config.getColorSpaceNames()
        except Exception as e:
            print(f"Warning: Could not get colorspaces: {e}")

        return ["auto"] + common + colorspaces

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_colorspace": (cls.get_colorspaces(), {"default": "sRGB"}),
                "output_colorspace": (cls.get_colorspaces(), {"default": DEFAULT_OUTPUT_TRANSFORM}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "oiio"

    def convert(self, image, input_colorspace="sRGB", output_colorspace=DEFAULT_OUTPUT_TRANSFORM):
        # No-op if colorspaces match or are set to auto
        if (
            input_colorspace == output_colorspace
            or input_colorspace == "auto"
            or output_colorspace == "auto"
        ):
            return (image,)

        # Ensure expected dims: [B,H,W,C] or [H,W,C]
        if image.dim() == 3:
            images = image.unsqueeze(0)
        elif image.dim() == 4:
            images = image
        else:
            raise ValueError(
                f"Unsupported image tensor dims {image.dim()} for shape {tuple(image.shape)}; expected [H,W,C] or [B,H,W,C]"
            )

        batch_size, height, width, channels = images.shape
        if channels not in (1, 3, 4):
            log.warning(
                f"Unexpected channel count in ColorspaceConvert: {channels}; proceeding but downstream nodes may fail"
            )

        log.debug(
            f"ColorspaceConvert: converting {batch_size} image(s) of size {width}x{height} with {channels} channels: {input_colorspace} -> {output_colorspace}"
        )

        converted_frames: list[torch.Tensor] = []
        for i in range(batch_size):
            frame = images[i].float().cpu().numpy()
            frame = np.ascontiguousarray(frame)

            # Build OIIO buffers (OIIO expects width, height, channels)
            in_spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
            roi = oiio.ROI(0, width, 0, height, 0, 1, 0, channels)
            in_buf = oiio.ImageBuf(in_spec)
            in_buf.set_pixels(roi, frame)

            out_buf = oiio.ImageBuf(in_spec)
            if not oiio.ImageBufAlgo.colorconvert(
                out_buf, in_buf, input_colorspace, output_colorspace
            ):
                raise RuntimeError(f"Color conversion failed: {out_buf.geterror()}")

            result_np = out_buf.get_pixels(roi=roi)
            result_t = torch.from_numpy(result_np).unsqueeze(0)

            converted_frames.append(result_t)

        result = torch.cat(converted_frames, dim=0)

        # Stats on first frame for quick sanity check
        try:
            _in_min, _in_max = images[0].min().item(), images[0].max().item()
            _out_min, _out_max = result[0].min().item(), result[0].max().item()
            log.info(
                f"Conversion stats - Input range: [{_in_min:.3f}, {_in_max:.3f}] | Output range: [{_out_min:.3f}, {_out_max:.3f}]"
            )
        except Exception:
            pass

        return (result,)



class OIIO_SaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        colorspaces = OIIO_ColorspaceConvert.get_colorspaces()
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "output.exr"}),
                "precision": (["half", "float"], {"default": "half"}),
                "compression": (
                    ["none", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"],
                    {"default": "zip"},
                ),
                "input_colorspace": (
                    colorspaces,
                    {"default": "sRGB"},
                ),
                "output_colorspace": (
                    colorspaces,
                    {"default": "sRGB"},
                ),
                "frame_start": ("INT", {"default": 1, "min": 0, "max": 999999}),
                "frame_pad": ("INT", {"default": 4, "min": 1, "max": 8}),
                "dwa_compression_level": ("FLOAT", {"default": 45.0, "min": 10.0, "max": 250.0}),
                "save_preview": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "write"
    CATEGORY = "oiio"
    EXPERIMENTAL = True
    OUTPUT_NODE = True

    def write(
        self,
        images,
        filename_prefix="output.exr",
        precision="half",
        compression="zip",
        input_colorspace="sRGB",
        output_colorspace="scene_linear",
        frame_start=1,
        frame_pad=4,
        dwa_compression_level=45.0,
        save_preview=True,
        prompt=None,
        extra_pnginfo=None,
    ):
        images = images.float()
        log.debug(
            f"Starting save with prefix: {filename_prefix}, precision: {precision}, colorspace: {input_colorspace} -> {output_colorspace}"
        )
        log.debug(f"Input tensor shape: {images.shape}, dtype: {images.dtype}")

        max_val = images.max().item()
        min_val = images.min().item()
        log.debug(f"Input tensor range: [{min_val:.6f}, {max_val:.6f}]")

        path = Path(filename_prefix)
        counter = 0
        if path.is_absolute():
            output_dir = path.parent
            filename = path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            log.debug(f"Using absolute path. Output dir: {output_dir}, filename: {filename}")
        else:
            c_out = folder_paths.get_output_directory()

            log.debug(f"Using Comfy output dir: {c_out}")
            stem_path = str(path).replace(path.suffix, "")
            log.debug(f"Path stem for get_save_image_path: {stem_path}")

            output_dir, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(stem_path, c_out, images.shape[2], images.shape[1])
            )
            log.debug(
                f"Got path components: dir={output_dir}, name={filename}, counter={counter}, subfolder={subfolder}"
            )

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        extension = path.suffix.lstrip(".")
        log.debug(f"Using extension: {extension}")
        last_path = None
        for i in range(images.shape[0]):
            if path.is_absolute():
                frame_num = frame_start + i
                output_path = output_dir / f"{filename}.{frame_num:0{frame_pad}d}.{extension}"
            else:
                output_path = output_dir / f"{filename}_{counter + i:0{frame_pad}d}.{extension}"

            log.debug(f"Processing frame {i}, output path: {output_path}")

            if i < 1:
                if prompt is not None:
                    prompt_path = output_path.with_stem(output_path.stem + "_api").with_suffix(
                        ".json"
                    )
                    log.debug(f"Writing prompt to: {prompt_path}")
                    with prompt_path.open("w") as f:
                        f.write(json.dumps(prompt, indent=2))
                if extra_pnginfo is not None:
                    info_path = output_path.with_stem(output_path.stem + "_ui").with_suffix(".json")
                    log.debug(f"Writing extra info to: {info_path}")
                    with info_path.open("w") as f:
                        for x in extra_pnginfo:
                            f.write(json.dumps(extra_pnginfo[x], indent=2))

            if precision == "float":
                pixels = images[i].float()
            elif precision == "half":
                pixels = images[i].half()
            else:
                pixels = images[i]

            log.debug(
                f"Tensor after precision conversion: dtype={pixels.dtype}, range=[{pixels.min().item():.6f}, {pixels.max().item():.6f}]"
            )

            pixels = pixels.cpu().numpy()
            pixels = np.ascontiguousarray(pixels)
            log.debug(
                f"Converted to numpy: shape={pixels.shape}, dtype={pixels.dtype}, range=[{pixels.min():.6f}, {pixels.max():.6f}]"
            )
            log.debug(f"Array flags: {pixels.flags}")

            if (
                input_colorspace != "auto"
                and output_colorspace != "auto"
                and input_colorspace != output_colorspace
            ):
                in_spec = ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], oiio.FLOAT)
                in_spec.attribute("oiio:ColorSpace", input_colorspace)
                in_buf = oiio.ImageBuf(in_spec)
                in_buf.set_pixels(
                    oiio.ROI(0, pixels.shape[1], 0, pixels.shape[0], 0, 1, 0, pixels.shape[2]),
                    pixels,
                )

                out_spec = ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], oiio.FLOAT)
                out_spec.attribute("oiio:ColorSpace", output_colorspace)
                out_buf = oiio.ImageBuf(out_spec)

                if not oiio.ImageBufAlgo.colorconvert(
                    out_buf, in_buf, input_colorspace, output_colorspace
                ):
                    log.error(f"Color conversion failed: {out_buf.geterror()}")
                else:
                    pixels = out_buf.get_pixels()

            spec = ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], precision)
            spec.attribute("compression", compression)
            if compression in ["dwaa", "dwab"]:
                spec.attribute("compressionlevel", dwa_compression_level)

            if output_colorspace != "auto":
                spec.set_colorspace(output_colorspace)

            if pixels.shape[2] == 1:
                spec.channelnames = ("A",)
            elif pixels.shape[2] == 3:
                spec.channelnames = ("R", "G", "B")
            elif pixels.shape[2] == 4:
                spec.channelnames = ("R", "G", "B", "A")

            log.debug(
                f"Image spec: {spec.width}x{spec.height}, {spec.nchannels} channels, format={spec.format}"
            )

            out = oiio.ImageOutput.create(str(output_path))
            if out:
                try:
                    log.debug(f"Opening file for writing: format={out.format_name()}")
                    success = out.open(str(output_path), spec)
                    if not success:
                        log.error(f"Failed to open file: {out.geterror()}")
                        continue

                    log.debug(
                        f"Pixel stats before write: shape={pixels.shape}, dtype={pixels.dtype}"
                    )
                    log.debug(
                        f"Pixel values - min={pixels.min():.6f}, max={pixels.max():.6f}, mean={pixels.mean():.6f}"
                    )
                    log.debug(
                        f"Unique values in first 1000: {np.unique(pixels.flatten()[:1000]).size}"
                    )

                    success = out.write_image(pixels)
                    if not success:
                        log.error(f"Failed to write image: {out.geterror()}")

                    log.debug(f"Successfully wrote image to: {output_path}")
                except Exception as e:
                    log.error(f"Failed to write image: {e}")
                finally:
                    out.close()
            else:
                log.error(f"Failed to create ImageOutput for: {output_path} {oiio.geterror()}")
            if i == 0 and save_preview:
                preview_path = output_path.with_suffix(".preview.png")
                preview_spec = ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], "uint8")
                preview_spec.attribute("oiio:ColorSpace", "sRGB")

                preview_out = oiio.ImageOutput.create(str(preview_path))
                if preview_out:
                    try:
                        preview_out.open(str(preview_path), preview_spec)
                        preview_pixels = pixels
                        if output_colorspace != "sRGB":
                            in_spec = ImageSpec(
                                pixels.shape[1], pixels.shape[0], pixels.shape[2], oiio.FLOAT
                            )
                            in_spec.attribute("oiio:ColorSpace", input_colorspace)
                            in_buf = oiio.ImageBuf(in_spec)
                            in_buf.set_pixels(
                                oiio.ROI(
                                    0, pixels.shape[1], 0, pixels.shape[0], 0, 1, 0, pixels.shape[2]
                                ),
                                pixels,
                            )

                            out_spec = ImageSpec(
                                pixels.shape[1], pixels.shape[0], pixels.shape[2], oiio.FLOAT
                            )
                            out_spec.attribute("oiio:ColorSpace", "sRGB")
                            out_buf = oiio.ImageBuf(out_spec)

                            oiio.ImageBufAlgo.colorconvert(
                                out_buf, in_buf, output_colorspace, "sRGB"
                            )
                            preview_pixels = out_buf.get_pixels()

                        preview_pixels = np.clip(preview_pixels * 255, 0, 255).astype(np.uint8)
                        preview_out.write_image(preview_pixels)
                    finally:
                        preview_out.close()
            last_path = output_path

        log.debug(f"Save completed. Last path: {last_path}")
        return (str(last_path),)


class OIIO_ColorspaceMatchFinder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "num_results": ("INT", {"default": 5, "min": 1, "max": 20}),
                "metrics": (["mse", "psnr", "both"], {"default": "both"}),
                "scale_factor": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 1.0, "step": 0.05}),
                "print_every": ("INT", {"default": 1000, "min": 100, "max": 10000}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("matches", "errors")
    FUNCTION = "find_matches"
    CATEGORY = "oiio"

    def compute_metrics(self, img1: torch.Tensor, img2: torch.Tensor) -> tuple[float, float]:
        mse = torch.mean((img1 - img2) ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")
        return mse, psnr

    def format_current_results(
        self, top_results: list[Any], metrics: str, tested: int, successful: int, failed: int
    ) -> str:
        lines = [
            f"\nInterim results after {tested} combinations (successful: {successful}, failed: {failed}):"
        ]

        results = sorted(top_results, key=lambda x: x[0])
        if metrics in ["psnr", "both"]:
            results.reverse()

        for i, (score, r) in enumerate(results):
            if metrics == "mse":
                lines.append(
                    f"{i + 1}. Input: {r['input_cs']}, Output: {r['output_cs']} (MSE: {r['mse']:.6f})"
                )
            elif metrics == "psnr":
                lines.append(
                    f"{i + 1}. Input: {r['input_cs']}, Output: {r['output_cs']} (PSNR: {r['psnr']:.2f}dB)"
                )
            else:
                lines.append(
                    f"{i + 1}. Input: {r['input_cs']}, Output: {r['output_cs']} (MSE: {r['mse']:.6f}, PSNR: {r['psnr']:.2f}dB)"
                )
        return "\n".join(lines)

    def find_matches(
        self, original_image, processed_image, num_results, metrics, scale_factor, print_every
    ):
        converter = OIIO_ColorspaceConvert()
        colorspaces = [cs for cs in converter.get_colorspaces() if cs != "auto"]
        error_counter: dict[str, Counter] = {}

        if scale_factor < 1.0:
            h = int(original_image.shape[1] * scale_factor)
            w = int(original_image.shape[2] * scale_factor)
            orig_buf, orig_roi = OIIOUtils.comfy_to_oiio(original_image)
            proc_buf, proc_roi = OIIOUtils.comfy_to_oiio(processed_image)
            scaled_orig = oiio.ImageBuf(oiio.ImageSpec(w, h, original_image.shape[3], oiio.FLOAT))
            scaled_proc = oiio.ImageBuf(oiio.ImageSpec(w, h, processed_image.shape[3], oiio.FLOAT))
            oiio.ImageBufAlgo.resize(scaled_orig, orig_buf)
            oiio.ImageBufAlgo.resize(scaled_proc, proc_buf)

            original = OIIOUtils.oiio_to_comfy(scaled_orig, scaled_orig.roi)
            processed = OIIOUtils.oiio_to_comfy(scaled_proc, scaled_proc.roi)

            del orig_buf, proc_buf, scaled_orig, scaled_proc
        else:
            original = original_image.float()
            processed = processed_image.float()

        total_combos = len(colorspaces) * len(colorspaces)
        tested = successful = failed = 0
        top_results = []

        proc_buf, proc_roi = OIIOUtils.comfy_to_oiio(processed)

        try:
            for in_cs, out_cs in itertools.product(colorspaces, colorspaces):
                tested += 1
                combo = f"{in_cs} -> {out_cs}"

                if tested % print_every == 0:
                    log.info(
                        self.format_current_results(
                            top_results, metrics, tested, successful, failed
                        )
                    )
                    gc.collect()
                try:
                    out_buf = OIIOUtils.convert_colorspace(proc_buf, in_cs, out_cs, proc_roi)
                    converted = OIIOUtils.oiio_to_comfy(out_buf, proc_roi)
                    mse, psnr = self.compute_metrics(original, converted)
                    if metrics == "mse":
                        score = -mse
                    elif metrics == "psnr":
                        score = psnr
                    else:
                        score = psnr / mse

                    entry = (
                        score,
                        {"input_cs": in_cs, "output_cs": out_cs, "mse": mse, "psnr": psnr},
                    )

                    if len(top_results) < num_results:
                        heapq.heappush(top_results, entry)
                    else:
                        if entry[0] > top_results[0][0]:
                            heapq.heapreplace(top_results, entry)

                    successful += 1
                except Exception as e:
                    failed += 1
                    error_msg = str(e)
                    if error_msg not in error_counter:
                        error_counter[error_msg] = Counter()
                    error_counter[error_msg][combo] += 1

                if "out_buf" in locals():
                    del out_buf

        finally:
            del proc_buf

        output_lines = [
            f"Best colorspace combinations (tested {tested}, successful: {successful}, failed: {failed}):"
        ]

        results = []
        while top_results:
            score, result = heapq.heappop(top_results)
            if metrics == "mse":
                # back to positive
                result["score"] = -score
            else:
                result["score"] = score
            results.append(result)

        for i, (score, r) in enumerate(results):
            if metrics == "mse":
                output_lines.append(
                    f"{i + 1}. Input: {r['input_cs']}, Output: {r['output_cs']} (MSE: {r['mse']:.6f})"
                )
            elif metrics == "psnr":
                output_lines.append(
                    f"{i + 1}. Input: {r['input_cs']}, Output: {r['output_cs']} (PSNR: {r['psnr']:.2f}dB)"
                )
            else:
                output_lines.append(
                    f"{i + 1}. Input: {r['input_cs']}, Output: {r['output_cs']} (MSE: {r['mse']:.6f}, PSNR: {r['psnr']:.2f}dB)"
                )

        error_lines = ["Colorspace conversion errors:"]
        for error_msg, combos in error_counter.items():
            error_lines.append(f"\nError: {error_msg}")
            error_lines.append(
                f"Occurred {sum(combos.values())} times with {len(combos)} different combinations"
            )
            error_lines.append("Top 5 most frequent combinations:")
            for combo, count in combos.most_common(5):
                error_lines.append(f"  {combo}: {count} times")

        return ("\n".join(output_lines), "\n".join(error_lines))


NODE_CLASS_MAPPINGS = {
    "OIIO_ColorspaceMatchFinder": OIIO_ColorspaceMatchFinder,
    "OIIO_LoadImage": OIIO_LoadImage,
    "OIIO_ColorspaceConvert": OIIO_ColorspaceConvert,
    "OIIO_SaveImage": OIIO_SaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OIIO_ColorspaceMatchFinder": "Colorspace Match Finder (OIIO)",
    "OIIO_ColorspaceConvert": "Colorspace Convert (OIIO)",
    "OIIO_LoadImage": "Load Image (OIIO)",
    "OIIO_SaveImage": "Save Image (OIIO)",
}
