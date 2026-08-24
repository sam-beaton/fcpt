"""
registerWangFunctional.py
==================
Transfers the Wang2021AgeCommon fine-grained functional parcellation (~430
parcels per hemisphere) from the UNC 4D Infant Cortical Surface Atlas V1.11
into the same NeuroDev volumetric space, producing a label NIfTI:
    {age}mo_WangFunctional_Reg_Head.nii.gz

This is a drop-in replacement for {age}mo_Parc_Reg_Head.nii.gz in the
NeuroDOT pipeline, with finer parcellation granularity.

Label convention
----------------
    LH parcels: IDs   1 – 430
    RH parcels: IDs 431 – 862  (pre-offset by the atlas — no additional
                                 offset applied by this script)

Pipeline per age
----------------
1. Read par_Wang2021AgeCommon vertex labels from atlas MiddleSurf VTK
   (left and right hemispheres separately)
2. Voxelise labels into UNC T1 atlas space (the 'moving' space).
   Surface vertex coordinates are in voxel index space, not world-mm.
3. Apply existing ANTs forward warp (UNC → NeuroDev) using NearestNeighbor
   interpolation — identical to how Parc_Reg_Head.nii.gz was produced
4. Save output NIfTI to OUTPUT_DIR/{age}mo/

Requirements
------------
    pip install pyvista nibabel numpy
    ANTs must be on PATH (set via os.environ in this script)

Usage
-----
    python registerWangFunctional.py

Edit the path variables in the CONFIG section below to match your system.
"""

import os
# NOTE: this PATH-prepending approach assumes a Mac/Linux-style ANTs install
# location and PATH separator (':'). ANTs isn't typically run natively on
# Windows - if you need this there, run it via WSL (which uses the same
# Mac/Linux-style path here) rather than trying to adapt this line directly.
os.environ['PATH'] = '/opt/ANTs/bin:' + os.environ['PATH']
import subprocess
import tempfile
import numpy as np
import nibabel as nib
import pyvista as pv

# =============================================================================
# CONFIG — edit these paths to match system
# =============================================================================

# Ages to process — must match atlas and warp file naming
AGES = ['01', '06', '12']

# Root directory of the UNC 4D surface atlas (VTK format folder)
# Expected structure: {ATLAS_VTK_DIR}/{age}mo/SparseAtlas_{age}_lh.MiddleSurf.vtk
ATLAS_VTK_DIR = '/Path/to/UNC-Infant-Cortical-Surface-Atlas_V1.11'

# UNC volumetric atlas T1s — used as the voxelisation reference space
# (the 'moving' image space before warping)
# Expected: {UNC_T1_DIR}/{age}Month/BCP-{age}M-T1.nii.gz
UNC_T1_DIR = '/Path/to//UNC-BCP_4D_Infant_Brain_Volumetric_Atlas'

# Directory containing ANTs warp files
# Expected: {WARP_DIR}/{age}mo/{age}mo_T1_UncVolumetric_to_NeuroDev1Warp.nii.gz
#           {WARP_DIR}/{age}mo/{age}mo_T1_UncVolumetric_to_NeuroDev0GenericAffine.mat
WARP_DIR = '/Path/to/ANTs/warp/files'

# NeuroDev reference image for antsApplyTransforms (the 'fixed' image)
# Expected: {NEURODEV_DIR}/Head/ANTS{age}-0Months3T_head_brain.nii.gz
NEURODEV_DIR = '/Path/to/NeuroDev/reference/image'

# Output directory — registered label volumes saved here
OUTPUT_DIR = '/Path/to/output/directory'

# Name of the Wang432 scalar array as stored in the VTK files.
# Run the inspection block at the bottom of this file if unsure.
WANG432_ARRAY_NAME = 'par_Wang2021AgeCommon'

# Voxel size (mm) for the intermediate label volume in atlas space.
# Match UNC T1 voxel size — 1mm isotropic is typical.
VOXEL_SIZE_MM = 1.0

# =============================================================================
# END CONFIG
# =============================================================================


def load_vtk_labels(vtk_path, array_name):
    """
    Load a surface VTK and return vertex coordinates and scalar label array.
    Raises a clear error if the array name is not found.
    """
    mesh = pv.read(vtk_path)
    if array_name not in mesh.point_data.keys():
        available = list(mesh.point_data.keys())
        raise KeyError(
            f"Array '{array_name}' not found in {vtk_path}.\n"
            f"Available arrays: {available}\n"
            f"Update WANG432_ARRAY_NAME in the CONFIG section."
        )
    coords = np.array(mesh.points)          # (N, 3) in atlas world-mm space
    labels = np.array(mesh.point_data[array_name]).astype(np.int32)
    return coords, labels


def voxelise_surface_labels(coords_lh, labels_lh,
                             coords_rh, labels_rh,
                             ref_nii_path,
                             rh_offset=430):
    """
    Rasterise surface vertex labels onto a voxel grid defined by ref_nii_path.

    Left hemisphere:  labels as-is (1–432)
    Right hemisphere: labels + rh_offset (433–864)

    Where multiple vertices map to the same voxel the most common label wins
    (mode vote). Background voxels are 0.
    """
    ref = nib.load(ref_nii_path)
    affine = ref.affine
    shape = ref.shape[:3]

    # World → voxel index transform
    inv_affine = np.linalg.inv(affine)

    label_vol = np.zeros(shape, dtype=np.int32)
    vote_count = np.zeros(shape, dtype=np.int32)  # track how many vertices hit each voxel

    def _rasterise(coords, labels, offset):
        # Apply offset to right hemisphere labels
        labs = labels + offset if offset > 0 else labels.copy()
        # Surface vertices are already in voxel index space — use directly
        vox_i = np.round(coords).astype(int)

        # Clip to volume bounds
        valid = (
            (vox_i[:, 0] >= 0) & (vox_i[:, 0] < shape[0]) &
            (vox_i[:, 1] >= 0) & (vox_i[:, 1] < shape[1]) &
            (vox_i[:, 2] >= 0) & (vox_i[:, 2] < shape[2]) &
            (labs > 0)          # skip background/unlabelled vertices
        )
        vox_i = vox_i[valid]
        labs  = labs[valid]

        # Mode vote: accumulate into a dict then write winners
        from collections import defaultdict
        vox_votes = defaultdict(lambda: defaultdict(int))
        for (i, j, k), lbl in zip(map(tuple, vox_i), labs):
            vox_votes[(i, j, k)][lbl] += 1

        for (i, j, k), vote_dict in vox_votes.items():
            winner = max(vote_dict, key=vote_dict.get)
            total  = sum(vote_dict.values())
            if total > vote_count[i, j, k]:
                label_vol[i, j, k] = winner
                vote_count[i, j, k] = total

    _rasterise(coords_lh, labels_lh, offset=0)
    _rasterise(coords_rh, labels_rh, offset=rh_offset)

    return nib.Nifti1Image(label_vol, affine, ref.header)


def apply_ants_warp(input_nii_path, ref_nii_path,
                    warp_path, affine_path,
                    output_nii_path):
    """
    Apply ANTs forward warp (UNC atlas → NeuroDev space) to a label volume.
    Uses NearestNeighbor interpolation to preserve integer labels exactly.
    """
    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', input_nii_path,
        '-r', ref_nii_path,
        '-o', output_nii_path,
        '-t', warp_path,
        '-t', affine_path,
        '-n', 'NearestNeighbor',
        '-v', '1',
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms failed:\n{result.stderr}"
        )
    print(f"  Done → {output_nii_path}")


def process_age(age):
    print(f"\n{'='*60}")
    print(f"Processing age: {age}mo")
    print(f"{'='*60}")

    # --- Paths ---
    lh_vtk = os.path.join(ATLAS_VTK_DIR, f'{age}',
                          f'SparseAtlas_{age}_lh.MiddleSurf.vtk')
    rh_vtk = os.path.join(ATLAS_VTK_DIR, f'{age}',
                          f'SparseAtlas_{age}_rh.MiddleSurf.vtk')
    unc_t1  = os.path.join(UNC_T1_DIR, f'{age}Month', f'BCP-{age}M-T1.nii.gz')
    warp    = os.path.join(WARP_DIR, f'{age}mo', f'{age}mo_T1_UncVolumetric_to_NeuroDev1Warp.nii.gz')
    affine  = os.path.join(WARP_DIR, f'{age}mo', f'{age}mo_T1_UncVolumetric_to_NeuroDev0GenericAffine.mat')
    ref     = os.path.join(NEURODEV_DIR, 'Head', f'ANTS{age}-0Months3T_head_brain.nii.gz')
    out_dir = os.path.join(OUTPUT_DIR, f'{age}mo')
    os.makedirs(out_dir, exist_ok=True)
    output = os.path.join(out_dir, f'{age}mo_WangFunctional_Reg_Head.nii.gz')

    # --- Check inputs exist ---
    for path, desc in [(lh_vtk, 'LH VTK'), (rh_vtk, 'RH VTK'),
                       (unc_t1, 'UNC T1'), (warp, 'ANTs warp'),
                       (affine, 'ANTs affine'), (ref, 'NeuroDev reference')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{desc} not found: {path}")

    # --- Step 1: Load surface labels ---
    print(f"  Loading LH labels from {lh_vtk}")
    coords_lh, labels_lh = load_vtk_labels(lh_vtk, WANG432_ARRAY_NAME)
    print(f"    {len(labels_lh)} vertices, labels {labels_lh.min()}–{labels_lh.max()}")

    print(f"  Loading RH labels from {rh_vtk}")
    coords_rh, labels_rh = load_vtk_labels(rh_vtk, WANG432_ARRAY_NAME)
    print(f"    {len(labels_rh)} vertices, labels {labels_rh.min()}–{labels_rh.max()} (no offset applied — atlas pre-offsets RH labels)")

    # --- Step 2: Voxelise into UNC T1 space ---
    print(f"  Voxelising labels into UNC T1 space ({unc_t1})")
    label_nii = voxelise_surface_labels(coords_lh, labels_lh,
                                        coords_rh, labels_rh,
                                        ref_nii_path=unc_t1,
                                        rh_offset=0)
    n_labelled = np.sum(label_nii.get_fdata() > 0)
    print(f"    {n_labelled} labelled voxels in atlas space")

    # --- Step 3: Save intermediate label volume to temp file ---
    with tempfile.NamedTemporaryFile(suffix='_wang432_atlas_space.nii.gz',
                                     delete=False) as tmp:
        tmp_path = tmp.name
    nib.save(label_nii, tmp_path)
    print(f"  Saved intermediate atlas-space labels to {tmp_path}")

    # --- Step 4: Apply ANTs warp ---
    print(f"  Applying ANTs warp to NeuroDev space...")
    apply_ants_warp(tmp_path, ref, warp, affine, output)

    # --- Cleanup ---
    os.unlink(tmp_path)

    # --- check output ---
    out_nii = nib.load(output)
    out_data = out_nii.get_fdata()
    unique_labels = np.unique(out_data[out_data > 0]).astype(int)
    print(f"Output: {output}")
    print(f"Unique parcel IDs in output: {len(unique_labels)} "
          f"(range {unique_labels.min()}–{unique_labels.max()})")
    print(f"LH parcels (1–430):   {np.sum(unique_labels <= 430)}")
    print(f"RH parcels (431–862): {np.sum(unique_labels > 430)}")


def inspect_vtk_arrays(age='01'):
    """
    Helper: print all scalar arrays available in the VTK file for a given age.
    Run this first if are unsure of the correct WANG432_ARRAY_NAME.
    Usage: change main() to call inspect_vtk_arrays() instead.
    """
    vtk_path = os.path.join(ATLAS_VTK_DIR, f'{age}',
                            f'SparseAtlas_{age}_lh.MiddleSurf.vtk')
    mesh = pv.read(vtk_path)
    print(f"\nArrays available in {vtk_path}:")
    for name in mesh.point_data.keys():
        arr = np.array(mesh.point_data[name])
        print(f"  '{name}': shape={arr.shape}, dtype={arr.dtype}, "
              f"unique values={len(np.unique(arr))}, "
              f"range={arr.min()}–{arr.max()}")


def main():
    # Uncomment the line below to inspect VTK array names before running:
    # inspect_vtk_arrays('01'); return

    for age in AGES:
        try:
            process_age(age)
        except Exception as e:
            print(f"\nERROR processing age {age}mo: {e}")
            raise

    print(f"\n{'='*60}")
    print("All ages complete.")
    print("Output files:")
    for age in AGES:
        out = os.path.join(OUTPUT_DIR, f'{age}mo', f'{age}mo_WangFunctional_Reg_Head.nii.gz')
        exists = os.path.exists(out)
        print(f"  {out}  {'[OK]' if exists else '[MISSING]'}")


if __name__ == '__main__':
    main()
