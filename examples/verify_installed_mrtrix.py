"""Opt-in, real-tool synthetic smoke test; no patient inputs or QC approval.

Run with: python examples/verify_installed_mrtrix.py --output-root /private/results
Requires installed MRtrix3. Uses one thread and fresh timestamped outputs.
The phantom is geometrical, not a realistic brain/FOD or ACT validation dataset.
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np

from dbs_connectome.external import (execute, filter_segment_crossings, audit_mask_exclusion,
                                    validate_sift2_weights, validate_tractogram_sampling)
from dbs_connectome.gradients import hemisphere_embeddings, load_nodes, validate_matrix, settings_dict
from dbs_connectome.provenance import new_run, write_json, freeze_inputs, assert_inputs_unchanged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    root = new_run(args.output_root, 'synthetic_mrtrix_prepared')
    source, output = root / 'synthetic_inputs', root / 'outputs'
    source.mkdir()
    output.mkdir()
    affine = np.diag([2., 2., 2., 1.])
    shape = (32, 32, 32)
    def save(name, array):
        path = source / (name + '.nii.gz')
        nib.save(nib.Nifti1Image(array, affine), path)
        return str(path)
    reference = save('reference', np.ones(shape, np.float32))
    mask = np.zeros(shape, np.uint8)
    mask[16, 16, 16] = 1
    mask_path = save('mask', mask)
    atlas = np.zeros(shape, np.int16)
    # Isotropic positive l=0 SH term: numerical interoperability only.
    fod = np.zeros((*shape, 6), np.float32)
    fod[..., 0] = 1
    fod_path = save('fod', fod)
    five = np.zeros((*shape, 5), np.float32)
    five[..., 2] = 1
    points, rows = [], []
    for hemi, x in [('L', 8), ('R', 23)]:
        for y in (5, 12, 19, 26):
            for z in (5, 12, 19, 26):
                i = len(points)
                points.append(np.asarray([x, y, z], dtype=float) * 2)
                atlas[x, y, z] = i + 1
                five[x, y, z, 2] = .2
                five[x, y, z, 0] = .8
                rows.append([i, i + 1, f'Synthetic_{hemi}_{i}', hemi, 'cortex', 'Synthetic'])
    atlas_path = save('atlas', atlas)
    five_path = save('five_tissue', five)
    nodes_path = source / 'nodes.tsv'
    with nodes_path.open('x', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['index', 'label', 'name', 'hemisphere', 'tissue', 'network'])
        writer.writerows(rows)
    tracks = []
    for offset in (0, 16):
        for i, j in combinations(range(offset, offset + 16), 2):
            a, b = points[i], points[j]
            count = int(np.ceil(np.linalg.norm(b - a) / .5)) + 1
            track = np.linspace(a, b, count, dtype=np.float32)
            tracks.extend([track] * (1 + (i * 7 + j * 3) % 5))
    expected_retained = len(tracks)
    tracks.append(np.linspace([30, 32, 32], [34, 32, 32], 9, dtype=np.float32))
    tracks.append(np.array([[30.8, 31.2, 32], [31.2, 30.8, 32]], dtype=np.float32))
    track_path = source / 'tracks.tck'
    nib.streamlines.save(nib.streamlines.Tractogram(tracks, affine_to_rasmm=np.eye(4)), str(track_path))
    before = freeze_inputs(source.iterdir())
    write_json(root / 'input_hashes_before.json', before)
    validate_tractogram_sampling(track_path, reference)
    sampled, final = output / 'mrtrix_excluded.tck', output / 'strict_excluded.tck'
    weights, mu, matrix_path = output / 'weights.txt', output / 'mu.txt', output / 'matrix.csv'
    exclusion = [['tckedit', str(track_path), str(sampled), '-exclude', mask_path, '-nthreads', '1']]
    fitting = [['tcksift2', str(final), fod_path, str(weights), '-act', five_path,
                '-out_mu', str(mu), '-nthreads', '1']]
    matrix_commands = [['tck2connectome', str(final), atlas_path, str(matrix_path),
                        '-tck_weights_in', str(weights), '-symmetric', '-zero_diagonal',
                        '-assignment_radial_search', '4', '-out_assignments', str(output / 'assignments.txt'),
                        '-nthreads', '1']]
    write_json(root / 'plan.json', exclusion + fitting + matrix_commands)
    for name, commands in [('exclude', exclusion), ('sift2', fitting), ('matrix', matrix_commands)]:
        directory = output / name
        directory.mkdir()
        if name == 'sift2':
            strict = filter_segment_crossings(sampled, mask_path, final)
            audit = audit_mask_exclusion(final, mask_path)
            assert audit['retained_streamlines'] == expected_retained
        if name == 'matrix':
            weight_audit = validate_sift2_weights(weights, mu, expected_retained)
        execute(commands, directory, 1)
    matrix = validate_matrix(np.loadtxt(matrix_path, delimiter=','))
    assert matrix.shape == (32, 32)
    assert np.allclose(matrix.sum() / 2, weight_audit['weight_sum'], rtol=1e-5)
    gradients = hemisphere_embeddings(matrix, load_nodes(nodes_path))
    np.savez_compressed(output / 'gradients.npz', **gradients)
    assert all(gradients[f'{h}_coordinates'].shape == (16, 10) for h in ('L', 'R'))
    assert_inputs_unchanged(before)
    result = {'synthetic_only': True, 'patient_data_used': False, 'human_qc_approval': False,
              'actual_tools': ['tckedit', 'tcksift2', 'tck2connectome', 'BrainSpace'],
              'tractography_generated_by': 'manual synthetic straight segments, not tckgen',
              'scope': 'prepared-input downstream interoperability; not DICOM, ACT generation or clinical validation',
              'input_tracks': len(tracks), 'strict_filter': strict, 'exclusion_audit': audit,
              'weight_audit': weight_audit, 'matrix_shape': list(matrix.shape),
              'matrix_weight_sum_checked': True, 'gradients': 'UNALIGNED_SINGLE_SESSION',
              'gradient_settings': settings_dict(), 'source_bytes_unchanged': True}
    write_json(root / 'result.json', result)
    print(f'Synthetic prepared-input smoke test passed: {root}', flush=True)


if __name__ == '__main__':
    main()
