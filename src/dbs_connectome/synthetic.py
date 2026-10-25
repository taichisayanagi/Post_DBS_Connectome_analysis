"""Input-free, geometrical integration fixture. Never accepts patient paths.

Known tissue/parcel maps replace FreeSurfer/NextBrain in this SMALL fixture.
No human QC approval is generated, and no clinical or anatomical validity is implied.
"""
import csv
import html
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np

from .external import (audit_mask_exclusion, check_threads, execute,
                       filter_segment_crossings, validate_sift2_weights,
                       validate_tractogram_sampling)
from .gradients import hemisphere_embeddings, load_nodes, settings_dict, validate_matrix
from .provenance import assert_inputs_unchanged, digest, freeze_inputs, record, write_json


TOOLS = ('dcm2niix', 'mrconvert', 'dwi2fod', '5tt2gmwmi', 'tckgen',
         'tckedit', 'tcksift2', 'tck2connectome')
NOTICE = ('SYNTHETIC GEOMETRY ONLY. No patient inputs, no human QC approval, '
          'no realistic anatomy, no clinical validation. Known atlas/5TT and a '
          'prescribed response replace anatomical segmentation/response estimation. '
          'No denoise, topup/eddy, registration estimation, FreeSurfer or NextBrain is run.')


def write_dicom(array, affine, directory, modality, series_number, study_uid, frame_uid):
    """Write an artificial, unambiguous 3-D CT/T1 series with explicit LPS geometry."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, MRImageStorage, generate_uid
    directory.mkdir()
    series_uid = generate_uid()
    storage = CTImageStorage if modality == 'CT' else MRImageStorage
    for z in range(array.shape[2]):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = storage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        path = directory / f'slice{z:03d}.dcm'
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
        ds.SOPClassUID, ds.SOPInstanceUID = storage, meta.MediaStorageSOPInstanceUID
        ds.PatientID, ds.PatientName = 'SYNTHETIC-ONLY', 'SYNTHETIC^GEOMETRY'
        ds.PatientBirthDate, ds.StudyDate, ds.StudyTime = '19000101', '20000101', '120000'
        ds.StudyInstanceUID, ds.FrameOfReferenceUID, ds.SeriesInstanceUID = study_uid, frame_uid, series_uid
        ds.SeriesNumber, ds.InstanceNumber = series_number, z+1
        ds.SeriesDescription, ds.Modality, ds.Manufacturer = 'SYNTHETIC_'+modality, modality, 'SYNTHETIC'
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        ds.Rows, ds.Columns = array.shape[1], array.shape[0]
        ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [2., 2.], 2., 2.
        ds.ImageOrientationPatient = [-1., 0., 0., 0., -1., 0.]
        ras = nib.affines.apply_affine(affine, [0, 0, z])
        ds.ImagePositionPatient = [-ras[0], -ras[1], ras[2]]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        if modality == 'MR':
            ds.MRAcquisitionType, ds.RepetitionTime, ds.EchoTime, ds.FlipAngle = '2D', 2000., 10., 90.
        else:
            ds.RescaleType = 'HU'
        ds.PixelData = array[:, :, z].T.astype('<i2').tobytes()
        ds.save_as(path, enforce_file_format=True)


def build_fixture(source):
    """Two balls with 16 surface sectors each; deliberately NOT a virtual patient."""
    from pydicom.uid import generate_uid
    source = Path(source)
    source.mkdir()
    shape = (44, 24, 24)
    grid = np.indices(shape).astype(float)
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = [-44., -24., -24.]
    brain = np.zeros(shape, bool)
    five = np.zeros((*shape, 5), np.float32)
    atlas = np.zeros(shape, np.int16)
    ct = np.zeros(shape, np.int16)
    rows = []
    for h, cx in enumerate((11., 32.)):
        delta = grid - np.array([cx, 12., 12.])[:, None, None, None]
        radius = np.sqrt((delta**2).sum(0))
        inside, grey = radius <= 9, (radius > 7) & (radius <= 9)
        brain |= inside
        five[..., 0][grey] = 1
        five[..., 2][radius <= 7] = 1
        angle = (np.arctan2(delta[1], delta[0])+2*np.pi) % (2*np.pi)
        azimuth = np.minimum((angle/(2*np.pi)*4).astype(int), 3)
        height = np.clip(((delta[2]/np.maximum(radius, 1)+1)*2).astype(int), 0, 3)
        sector = height*4 + azimuth
        atlas[grey] = (sector[grey]+1+h*16).astype(np.int16)
        ct[(delta[0] == 0) & (delta[1] == 0) & (np.abs(delta[2]) <= 8)] = 3000
        for i in range(16):
            rows.append([h*16+i, h*16+i+1, f'Synthetic_{"L" if h == 0 else "R"}_{i+1}',
                         'L' if h == 0 else 'R', 'cortex', 'Synthetic'])
    t1 = (five[..., 2]*1000 + five[..., 0]*700).astype(np.int16)
    b0 = t1.astype(np.float32)
    b0[ct > 0] = 0  # Known synthetic signal dropout, not a susceptibility simulation.
    paths = {}
    for name, data in [('T1', t1), ('CT', ct), ('b0', b0), ('brain', brain.astype(np.uint8)),
                       ('atlas', atlas), ('five_tissue', five)]:
        path = source / (name+'.nii.gz')
        nib.save(nib.Nifti1Image(data, affine), path)
        paths[name] = str(path)
    # Uniform directional signal: tests CSD interoperability, NOT a biological diffusion model.
    count = 64
    z = 1-2*(np.arange(count)+.5)/count
    theta = np.arange(count)*np.pi*(3-np.sqrt(5))
    directions = np.array([np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z])
    bvec = np.column_stack([np.zeros((3, 4)), directions])
    bval = np.r_[np.zeros(4), np.full(count, 1000.)]
    dwi = np.stack([t1 * (1. if b == 0 else .5) for b in bval], axis=-1).astype(np.float32)
    nib.save(nib.Nifti1Image(dwi, affine), source/'dwi.nii.gz')
    np.savetxt(source/'dwi.bvec', bvec)
    np.savetxt(source/'dwi.bval', bval[None])
    np.savetxt(source/'response.txt', [[1000., -300., 100.]])
    write_json(source/'dwi.json', {'Synthetic': True, 'Notice': NOTICE})
    with (source/'nodes.tsv').open('x', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['index', 'label', 'name', 'hemisphere', 'tissue', 'network'])
        writer.writerows(rows)
    uid, frame = generate_uid(), generate_uid()
    write_dicom(t1, affine, source/'dicom_t1', 'MR', 1, uid, frame)
    write_dicom(ct, affine, source/'dicom_ct', 'CT', 2, uid, frame)
    write_json(source/'SYNTHETIC_ONLY.json', {'synthetic_only': True, 'notice': NOTICE,
                                            'shape': list(shape), 'node_count': 32})
    return paths


def verify_dicom_conversion(original, converted):
    a, b = [nib.as_closest_canonical(nib.load(p)) for p in (original, converted)]
    if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-5):
        raise ValueError('Synthetic DICOM conversion changed world-space geometry')
    if not np.array_equal(a.get_fdata(), b.get_fdata()):
        raise ValueError('Synthetic DICOM conversion changed decoded values')


def _csv(path, header, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def verify_matrix_weights(matrix, assignments_path, weights_path):
    """Independent endpoint-weight sum, including discarded self/unassigned tracks."""
    pairs = np.loadtxt(assignments_path, dtype=int, ndmin=2)
    weights = np.loadtxt(weights_path, ndmin=1)
    if pairs.shape != (len(weights), 2) or np.any(pairs < 0) or np.any(pairs > len(matrix)):
        raise ValueError('Unexpected endpoint assignments')
    expected = np.zeros_like(matrix)
    used = (pairs[:, 0] > 0) & (pairs[:, 1] > 0) & (pairs[:, 0] != pairs[:, 1])
    i, j = pairs[used].T-1
    np.add.at(expected, (i,j), weights[used])
    np.add.at(expected, (j,i), weights[used])
    if not np.allclose(matrix, expected, rtol=1e-5, atol=1e-7):
        raise ValueError('Weighted matrix differs from independently summed assignments')
    return {'max_absolute_error':float(np.max(np.abs(matrix-expected))),
            'assigned_nonself_tracks':int(used.sum()), 'discarded_self_or_unassigned_tracks':int((~used).sum()),
            'assigned_weight_sum':float(weights[used].sum()), 'matrix_upper_triangle_sum':float(matrix.sum()/2)}


def run_synthetic(out, threads=2, execute_tools=False):
    """Fixed input-free route; cannot bypass the clinical workflow QC gates."""
    from .electrodes import make_electrode
    from .graphs import graph_features
    from .qc import browser_report
    started = time.monotonic()
    check_threads(threads)
    out = Path(out).resolve()
    if execute_tools:
        from .external import command_environment
        missing = [name for name in TOOLS if shutil.which(name, path=command_environment(threads)['PATH']) is None]
        if missing:
            raise ValueError('Synthetic demo requires installed tools: '+', '.join(missing))
    source = out/'synthetic_inputs'
    p = build_fixture(source)
    before = freeze_inputs(x for x in source.rglob('*') if x.is_file())
    write_json(out/'input_hashes.json', before)
    for name in ('converted', 'reconstruction', 'electrode', 'tractography', 'connectome', 'gradient'):
        (out/name).mkdir()
    conversion = [['dcm2niix', '-z', 'y', '-b', 'y', '-f', f'synthetic_{role}',
                   '-o', str(out/'converted'), str(source/f'dicom_{role}')] for role in ('t1', 'ct')]
    fod = out/'reconstruction/fod.mif'
    reconstruction = [
        ['mrconvert', str(source/'dwi.nii.gz'), str(out/'reconstruction/dwi.mif'),
         '-fslgrad', str(source/'dwi.bvec'), str(source/'dwi.bval'), '-nthreads', str(threads)],
        ['dwi2fod', 'csd', str(out/'reconstruction/dwi.mif'), str(source/'response.txt'), str(fod),
         '-mask', p['brain'], '-lmax', '4', '-nthreads', str(threads)]]
    tracks, sampled, final = [out/'tractography'/name for name in ('tracks.tck', 'sampled_excluded.tck', 'strict_excluded.tck')]
    gmwmi, mask = out/'tractography/gmwmi.mif', out/'electrode/mask.nii.gz'
    tracking = [
        ['5tt2gmwmi', p['five_tissue'], str(gmwmi), '-nthreads', str(threads)],
        ['tckgen', str(fod), str(tracks), '-algorithm', 'iFOD2', '-act', p['five_tissue'],
         '-backtrack', '-seed_gmwmi', str(gmwmi), '-select', '10000', '-seeds', '1000000',
         '-cutoff', '0.06', '-step', '0.5', '-angle', '45', '-minlength', '4', '-maxlength', '80',
         '-downsample', '1', '-nthreads', str(threads)]]
    exclusion = [['tckedit', str(tracks), str(sampled), '-exclude', str(mask), '-nthreads', str(threads)]]
    weights, mu, matrix_path = [out/'connectome'/name for name in ('weights.txt', 'mu.txt', 'matrix.csv')]
    fitting = [['tcksift2', str(final), str(fod), str(weights), '-act', p['five_tissue'],
                '-out_mu', str(mu), '-nthreads', str(threads)]]
    connectome = [['tck2connectome', str(final), p['atlas'], str(matrix_path), '-tck_weights_in', str(weights),
                   '-symmetric', '-zero_diagonal', '-assignment_radial_search', '4',
                   '-out_assignments', str(out/'connectome/assignments.txt'), '-nthreads', str(threads)]]
    write_json(out/'plan.json', {'notice': NOTICE, 'commands': conversion+reconstruction+tracking+exclusion+fitting+connectome,
                                'synthetic_only': True, 'no_human_qc_approval_created': True})
    if not execute_tools:
        record(out, 'synthetic-demo', list(before), {'notice': NOTICE}, [out/'plan.json'], 'planned_not_executed')
        return
    def call(name, commands):
        directory = out/(name+'_commands')
        directory.mkdir()
        execute(commands, directory, threads)
    print('[step 1/8] Synthetic CT/T1 DICOM conversion and exact geometry check', flush=True)
    call('dicom', conversion)
    for role in ('t1', 'ct'):
        verify_dicom_conversion(p[role.upper()], out/f'converted/synthetic_{role}.nii.gz')
        # Lossless axis permutation/flip only, not anatomical registration estimation.
        nib.save(nib.as_closest_canonical(nib.load(out/f'converted/synthetic_{role}.nii.gz')),
                 out/f'converted/synthetic_{role}_ras.nii.gz')
    print('[step 2/8] NIfTI diffusion input -> CSD using prescribed response', flush=True)
    call('fod', reconstruction)
    print('[step 3/8] CT + MRI-void candidate on known identical synthetic grids', flush=True)
    image_paths = {'reference': p['b0'], 'CT': str(out/'converted/synthetic_ct_ras.nii.gz'), 'MRI': p['b0'], 'Brain': p['brain']}
    # The fixture's geometry/HU are known by construction. This is not patient sign-off.
    spec = {'coordinate_system': 'NIFTI_RAS_MM', 'reference_sha256': digest(p['b0']),
            'image_hashes': {k:digest(v) for k,v in image_paths.items()}, 'mode':'ct',
            'registration_reviewed': True, 'ct_hu_reviewed': True, 'expected_leads':2,
            'radius_mm': 1., 'corridor_mm': 3., 'dilation_passes':1, 'include_mri_void':True,
            'synthetic_fixture_only':True, 'review_basis':'analytic construction, NOT human clinical QC'}
    electrode = make_electrode(p['b0'], spec, out/'electrode', ct=image_paths['CT'], mri=p['b0'], brain_mask=p['brain'])
    binary = nib.load(mask).get_fdata() > 0
    if not np.all(binary[nib.load(p['CT']).get_fdata() >= 2000]):
        raise ValueError('Synthetic mask failed to cover the known metal voxels')
    browser_report(p['b0'], mask, out/'mask_review.html', atlas=p['atlas'],
                   backgrounds={'CT':image_paths['CT'], 'T1':p['T1']}, five_tissue=p['five_tissue'])
    print('[step 4/8] Actual ACT/iFOD2 generation: 10000 streamlines', flush=True)
    call('act', tracking)
    input_tracks = validate_tractogram_sampling(tracks, p['b0'])
    if input_tracks['count'] != 10000:
        raise ValueError('ACT did not produce the requested synthetic streamline count')
    print('[step 5/8] Electrode exclusion and independent continuous-segment audit', flush=True)
    call('exclusion', exclusion)
    strict = filter_segment_crossings(sampled, mask, final)
    audit = audit_mask_exclusion(final, mask)
    retained = audit['retained_streamlines']
    if retained < 1000:
        raise ValueError('Too few retained synthetic tracks')
    print('[step 6/8] SIFT2 refit after exclusion -> weighted connectome', flush=True)
    call('sift2', fitting)
    weight_audit = validate_sift2_weights(weights, mu, retained)
    call('matrix', connectome)
    matrix = validate_matrix(np.loadtxt(matrix_path, delimiter=','))
    if matrix.shape != (32, 32) or np.any(matrix.sum(1) <= 0):
        raise ValueError('Synthetic connectome lost a parcel; do not silently pad it')
    matrix_audit = verify_matrix_weights(matrix, out/'connectome/assignments.txt', weights)
    np.save(out/'connectome/connectome.npy', matrix)
    print('[step 7/8] Hemisphere gradients G1-G4; graph/node/edge exports', flush=True)
    nodes = load_nodes(source/'nodes.tsv')
    gradients = hemisphere_embeddings(matrix, nodes)
    np.savez_compressed(out/'gradient/gradients.npz', **gradients)
    _csv(out/'gradient/G1_G4.csv', ['node','hemisphere','G1','G2','G3','G4'],
         [[nodes[int(i)]['name'], h, *g[:4]] for h in ('L','R')
          for i,g in zip(gradients[h+'_indices'], gradients[h+'_coordinates'])])
    _csv(out/'connectome/nodes.csv', ['node','strength','degree'],
         [[n['name'],float(matrix[i].sum()),int(np.count_nonzero(matrix[i]))] for i,n in enumerate(nodes)])
    _csv(out/'connectome/edges.csv', ['source','target','weight'],
         [[nodes[i]['name'],nodes[j]['name'],float(matrix[i,j])] for i,j in zip(*np.triu_indices(32,1)) if matrix[i,j]>0])
    graph = graph_features(matrix, costs=(.10,.20,.30), modularity=False)
    graph = {k:(float(v) if np.isfinite(v) else None) for k,v in graph.items()}
    write_json(out/'connectome/graph.json', graph)
    assert_inputs_unchanged(before)
    result = {'status':'completed_synthetic_integration', 'synthetic_only':True, 'patient_data_used':False,
              'human_qc_approval':False, 'full_image_to_paper_validated':False, 'notice':NOTICE,
              'elapsed_seconds':round(time.monotonic()-started,2), 'threads':threads,
              'dicom_value_and_geometry_checks':2, 'actual_tools':list(TOOLS),
              'streamlines_requested':10000, 'sampling':input_tracks, 'retained_streamlines':retained,
              'strict_filter':strict, 'exclusion_audit':audit, 'weight_audit':weight_audit, 'matrix_weight_audit':matrix_audit,
              'matrix_shape':list(matrix.shape), 'nonzero_edges':int(np.count_nonzero(np.triu(matrix,1))),
              'mask_covers_all_synthetic_metal':True, 'source_bytes_unchanged':True,
              'atlas_and_5tt':'known synthetic truth, NOT FreeSurfer/NextBrain segmentation',
              'gradient_settings':settings_dict(), 'gradient_alignment':'UNALIGNED_SINGLE_SESSION',
              'tractography_seed':'stochastic; structural checks, not bitwise tractogram replay',
              'electrode_automatic_qc_approval':electrode['automatic_qc_approval']}
    write_json(out/'synthetic_result.json', result)
    _report(out, result, graph)
    record(out, 'synthetic-demo', list(before), {'threads':threads, 'notice':NOTICE},
           [out/'synthetic_result.json', out/'synthetic_review.html', out/'mask_review.html',
            out/'connectome/connectome.npy', out/'gradient/gradients.npz', out/'gradient/G1_G4.csv'])
    print('[step 8/8] Synthetic integration checks passed; no patient validation claimed', flush=True)


def _report(out, result, graph):
    rows = [('Status',result['status']), ('Elapsed',str(result['elapsed_seconds'])+' s'),
            ('DICOM exact voxel + world-coordinate checks','2 / 2'), ('ACT streamlines requested','10000'),
            ('Retained after exclusion',str(result['retained_streamlines'])),
            ('Weighted matrix','32 x 32'), ('Nonzero edges',str(result['nonzero_edges'])),
            ('Endpoint / SIFT2 independent weight check','Passed'),
            ('Gradients','G1-G4 exported; separate hemispheres; unaligned single session'),
            ('Original synthetic inputs','Unchanged'), ('Patient data / clinical QC approval','None')]
    table = ''.join('<tr><th>'+html.escape(k)+'</th><td>'+html.escape(v)+'</td></tr>' for k,v in rows)
    gtable = ''.join('<tr><th>'+html.escape(k)+'</th><td>'+('undefined' if v is None else f'{v:.6g}')+'</td></tr>'
                     for k,v in graph.items() if k.endswith('cost_mean'))
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            '<title>Synthetic integration report</title><style>body{font:15px system-ui;margin:32px;max-width:1100px;'
            'background:white;color:#1f2933}table{border-collapse:collapse;width:100%}th,td{text-align:left;'
            'padding:8px;border-bottom:1px solid #ddd}p{line-height:1.6}.notice{border-left:4px solid #986500;padding:12px}</style>'
            '<h1>Synthetic integration report</h1><p class="notice">'+html.escape(NOTICE)+'</p><table>'+table+'</table>'
            '<h2>Executed route</h2><p>CT/T1 DICOM conversion + NIfTI DWI → CSD → CT/MRI mask → '
            'ACT/iFOD2 → electrode exclusion → SIFT2 refit → connectome → G1–G4 / graph / node / edge.</p>'
            '<h2>Not tested by this fixture</h2><p>Realistic anatomy, susceptibility correction, registration estimation, '
            'FreeSurfer/NextBrain, human QC, longitudinal clinical associations, named tract segmentation and FEM/VTA. '
            'No scientific result is inferred from these artificial values.</p><h2>Graph output (artificial units)</h2><table>'+gtable+'</table></html>')
    (out/'synthetic_review.html').write_text(page, encoding='utf-8')
