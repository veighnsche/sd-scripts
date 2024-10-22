import argparse
import os
import json
import random

from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from library.device_utils import init_ipex, get_preferred_device

init_ipex()

from torchvision import transforms

import library.model_util as model_util
import library.train_util as train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

DEVICE = get_preferred_device()

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Import necessary libraries for dynamic bucketing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_npz_filename(data_dir, image_key, is_full_path, recursive):
    if is_full_path:
        base_name = os.path.splitext(os.path.basename(image_key))[0]
        relative_path = os.path.relpath(os.path.dirname(image_key), data_dir)
    else:
        base_name = image_key
        relative_path = ""

    if recursive and relative_path:
        return os.path.join(data_dir, relative_path, base_name) + ".npz"
    else:
        return os.path.join(data_dir, base_name) + ".npz"


def cluster_images(
    image_paths, num_buckets, min_images_per_bucket, bucket_dimension_multiple
):
    """Clusters images into buckets based on their dimensions and aspect ratios."""

    widths = []
    heights = []
    aspect_ratios = []
    valid_image_paths = []

    # Collect image data
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width == 0 or height == 0:
                    logger.warning(
                        f"Warning: Image {image_path} has zero width or height. Skipping."
                    )
                    continue
                valid_image_paths.append(image_path)
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            continue

    if len(valid_image_paths) == 0:
        logger.error("No valid images found in the input folder.")
        return None, None, None

    # Prepare data for clustering
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    X = np.column_stack((widths, heights, aspect_ratios))

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initial Clustering
    kmeans = KMeans(n_clusters=num_buckets, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Handle small clusters
    cluster_counts = np.bincount(clusters)
    small_clusters = np.where(cluster_counts < min_images_per_bucket)[0]

    if len(small_clusters) > 0:
        logger.info(
            f"Found {len(small_clusters)} clusters with fewer than {min_images_per_bucket} images. Merging them with nearest clusters."
        )
        for small_cluster in small_clusters:
            # Indices of images in the small cluster
            small_indices = np.where(clusters == small_cluster)[0]
            # For each image in the small cluster
            for idx in small_indices:
                # Compute distances to all cluster centers
                distances = np.linalg.norm(
                    X_scaled[idx] - kmeans.cluster_centers_, axis=1
                )
                # Exclude the small cluster itself
                distances[small_cluster] = np.inf
                # Assign to the nearest cluster
                new_cluster = np.argmin(distances)
                clusters[idx] = new_cluster
        # Recompute cluster counts after reassignment
        cluster_counts = np.bincount(clusters, minlength=num_buckets)

    # Remove empty clusters
    unique_clusters = np.unique(clusters)
    cluster_mapping = {
        old_label: new_label for new_label, old_label in enumerate(unique_clusters)
    }
    clusters = np.array([cluster_mapping[old_label] for old_label in clusters])
    num_buckets = len(unique_clusters)

    # Determine bucket dimensions
    buckets = {}
    for i in range(num_buckets):
        indices = np.where(clusters == i)[0]
        if len(indices) == 0:
            continue  # Skip empty clusters
        cluster_widths = widths[indices]
        cluster_heights = heights[indices]
        # Calculate the mean width and height
        mean_width = np.mean(cluster_widths)
        mean_height = np.mean(cluster_heights)
        # Adjust to nearest specified pixel steps
        bucket_width = int(
            round(mean_width / bucket_dimension_multiple) * bucket_dimension_multiple
        )
        bucket_height = int(
            round(mean_height / bucket_dimension_multiple) * bucket_dimension_multiple
        )
        buckets[i] = {
            "bucket_width": bucket_width,
            "bucket_height": bucket_height,
            "image_indices": indices,
        }

    # Create a mapping from image path to cluster index
    image_to_cluster = {
        valid_image_paths[i]: clusters[i] for i in range(len(valid_image_paths))
    }

    return buckets, image_to_cluster, valid_image_paths


def resize_and_crop_image(img, bucket_width, bucket_height, crop_mode):
    """Resizes and crops the image to fit into the bucket dimensions."""

    orig_width, orig_height = img.size

    # Calculate scaling factors
    scale_width = bucket_width / orig_width
    scale_height = bucket_height / orig_height
    # Choose the larger scaling factor to ensure the image fills the bucket dimensions
    scale = max(scale_width, scale_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Calculate excess pixels
    delta_width = new_width - bucket_width
    delta_height = new_height - bucket_height

    # Initialize cropping parameters
    crop_x = 0
    crop_y = 0

    # Determine cropping along one axis
    if delta_width >= delta_height:
        # Crop along width (x-axis)
        if crop_mode == "center":
            crop_x = (new_width - bucket_width) // 2
        else:
            crop_x = random.randint(0, new_width - bucket_width)
        crop_box = (crop_x, 0, crop_x + bucket_width, new_height)
    else:
        # Crop along height (y-axis)
        if crop_mode == "center":
            crop_y = (new_height - bucket_height) // 2
        else:
            crop_y = random.randint(0, new_height - bucket_height)
        crop_box = (0, crop_y, new_width, crop_y + bucket_height)

    cropped_img = resized_img.crop(crop_box)

    return cropped_img


def main(args):
    if args.bucket_reso_steps % 8 > 0:
        logger.warning(
            f"Resolution of buckets in training time is a multiple of 8."
        )
    if args.bucket_reso_steps % 32 > 0:
        logger.warning(
            f"WARNING: bucket_reso_steps is not divisible by 32. It may not work with SDXL."
        )

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[str] = [
        str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    ]
    logger.info(f"Found {len(image_paths)} images.")

    if os.path.exists(args.in_json):
        logger.info(f"Loading existing metadata: {args.in_json}")
        with open(args.in_json, "rt", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.error(f"No metadata file found: {args.in_json}")
        return

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = model_util.load_vae(args.model_name_or_path, weight_dtype)
    vae.eval()
    vae.to(DEVICE, dtype=weight_dtype)

    # Initialize BucketManager
    if args.use_dynamic_buckets:
        # Dynamic bucketing using KMeans clustering
        logger.info("Using dynamic buckets based on image dimensions.")
        buckets, image_to_cluster, valid_image_paths = cluster_images(
            image_paths,
            args.num_buckets,
            args.min_images_per_bucket,
            args.bucket_dimension_multiple,
        )
        if buckets is None:
            logger.error("Failed to create buckets.")
            return
        # Update image_paths to only include valid images
        image_paths = valid_image_paths
    else:
        # Predefined bucket sizes
        max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
        assert len(max_reso) == 2, (
            f"Illegal resolution (not 'width,height'): {args.max_resolution}"
        )

        bucket_manager = train_util.BucketManager(
            args.bucket_no_upscale,
            max_reso,
            args.min_bucket_reso,
            args.max_bucket_reso,
            args.bucket_reso_steps,
        )
        if not args.bucket_no_upscale:
            bucket_manager.make_buckets()
        else:
            logger.warning(
                "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, "
                "because bucket reso is defined by image size automatically."
            )

    img_ar_errors = []

    def process_batch(is_last):
        if args.use_dynamic_buckets:
            # No batching needed here
            pass
        else:
            for bucket in bucket_manager.buckets:
                if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
                    train_util.cache_batch_latents(
                        vae, True, bucket, args.flip_aug, args.alpha_mask, False
                    )
                    bucket.clear()

    # Use DataLoader if specified
    if args.max_data_loader_n_workers is not None:
        dataset = train_util.ImageLoadingDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    bucket_counts = {}
    for data_entry in tqdm(data, smoothing=0.0):
        if data_entry[0] is None:
            continue

        img_tensor, image_path = data_entry[0]
        if img_tensor is not None:
            image = transforms.functional.to_pil_image(img_tensor)
        else:
            try:
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception as e:
                logger.error(f"Could not load image path: {image_path}, error: {e}")
                continue

        image_key = (
            image_path
            if args.full_path
            else os.path.splitext(os.path.basename(image_path))[0]
        )
        if image_key not in metadata:
            metadata[image_key] = {}

        if args.use_dynamic_buckets:
            if image_path not in image_to_cluster:
                logger.warning(f"Image {image_path} was not clustered. Skipping.")
                continue
            # Assign image to bucket
            bucket_idx = image_to_cluster[image_path]
            bucket = buckets[bucket_idx]
            bucket_width = bucket["bucket_width"]
            bucket_height = bucket["bucket_height"]
            reso = (bucket_width, bucket_height)
            resized_size = reso
            ar_error = 0  # Aspect ratio error is zero since we fit images to bucket dimensions

            # Resize and crop the image
            image = resize_and_crop_image(
                image, bucket_width, bucket_height, args.crop_mode
            )

            # Log resizing and cropping information if needed
            # logger.info(f"Processed image {image_key} into bucket {reso}")
        else:
            # Existing bucket assignment
            reso, resized_size, ar_error = bucket_manager.select_bucket(
                image.width, image.height
            )

        img_ar_errors.append(abs(ar_error))
        bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

        # Record training resolution in metadata (multiple of 8)
        metadata[image_key]["train_resolution"] = (
            reso[0] - reso[0] % 8,
            reso[1] - reso[1] % 8,
        )

        # Check if latent files already exist
        npz_file_name = get_npz_filename(
            args.train_data_dir, image_key, args.full_path, args.recursive
        )
        if args.skip_existing:
            if train_util.is_disk_cached_latents_is_expected(
                reso, npz_file_name, args.flip_aug
            ):
                continue

        # Prepare image info
        image_info = train_util.ImageInfo(
            image_key, 1, "", False, image_path
        )
        image_info.latents_npz = npz_file_name
        image_info.bucket_reso = reso
        image_info.resized_size = resized_size
        image_info.image = image

        if args.use_dynamic_buckets:
            # Directly process the image
            train_util.cache_latents(
                vae, True, image_info, args.flip_aug, args.alpha_mask, False
            )
        else:
            # Add image to bucket for batch processing
            bucket_manager.add_image(reso, image_info)

        # Process batch if necessary
        process_batch(False)

    # Process any remaining images
    process_batch(True)

    if not args.use_dynamic_buckets:
        bucket_manager.sort()
        for i, reso in enumerate(bucket_manager.resos):
            count = bucket_counts.get(reso, 0)
            if count > 0:
                logger.info(f"Bucket {i} {reso}: {count} images")
    else:
        # Log bucket counts for dynamic buckets
        logger.info("Bucket dimensions and image counts:")
        for reso, count in bucket_counts.items():
            logger.info(f"Bucket {reso}: {count} images")

    if len(img_ar_errors) > 0:
        img_ar_errors = np.array(img_ar_errors)
        logger.info(f"Mean aspect ratio error: {np.mean(img_ar_errors)}")
    else:
        logger.info("No images were processed. Aspect ratio error cannot be calculated.")

    # Write metadata
    logger.info(f"Writing metadata: {args.out_json}")
    with open(args.out_json, "wt", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir",
        type=str,
        help="Directory for train images",
    )
    parser.add_argument(
        "in_json",
        type=str,
        help="Metadata file to input",
    )
    parser.add_argument(
        "out_json",
        type=str,
        help="Metadata file to output",
    )
    parser.add_argument(
        "model_name_or_path",
        type=str,
        help="Model name or path to encode latents",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Not used (for backward compatibility)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size in inference",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="Enable image reading by DataLoader with this number of workers (faster)",
    )
    parser.add_argument(
        "--max_resolution",
        type=str,
        default="512,512",
        help="Max resolution in fine-tuning (width,height)",
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=256,
        help="Minimum resolution for buckets",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=1024,
        help="Maximum resolution for buckets",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="Steps of resolution for buckets, divisible by 8 is recommended",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="Make bucket for each image without upscaling",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Use mixed precision",
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="Use full path as image-key in metadata (supports multiple directories)",
    )
    parser.add_argument(
        "--flip_aug",
        action="store_true",
        help="Flip augmentation, save latents for flipped images",
    )
    parser.add_argument(
        "--alpha_mask",
        type=str,
        default="",
        help="Save alpha mask for images for loss calculation",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images if npz already exists (both normal and flipped exist if flip_aug is enabled)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively look for training tags in all child folders of train_data_dir",
    )
    # Additional arguments for dynamic bucketing
    parser.add_argument(
        "--use_dynamic_buckets",
        action="store_true",
        help="Use dynamic buckets based on image dimensions",
    )
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=5,
        help="Number of buckets for dynamic bucketing",
    )
    parser.add_argument(
        "--min_images_per_bucket",
        type=int,
        default=6,
        help="Minimum number of images per bucket before merging (default: 6)",
    )
    parser.add_argument(
        "--bucket_dimension_multiple",
        type=int,
        default=64,
        help="Multiple to which bucket dimensions are adjusted (default: 64)",
    )
    parser.add_argument(
        "--crop_mode",
        choices=["center", "random"],
        default="center",
        help="Crop mode: 'center' or 'random' (default: 'center')",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
