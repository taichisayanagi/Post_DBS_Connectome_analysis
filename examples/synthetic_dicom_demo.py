"""Small, non-anatomical DICOM import fixture. NOT a diffusion validation dataset."""

import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from dbs_connectome.provenance import new_run, write_json


def build(root):
    out = new_run(root, "synthetic_dicom_import")
    study, frame = generate_uid(), generate_uid()
    for number, description in ((1, "SYNTHETIC T1 phantom"), (2, "SYNTHETIC DWI placeholder NOT diffusion")):
        directory = out / f"series{number}"
        directory.mkdir(mode=0o700)
        series = generate_uid()
        for z in range(12):
            path = directory / f"slice{z:03d}.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.PatientID = "SYNTHETIC-ONLY"
            ds.PatientName = "SYNTHETIC^PHANTOM"
            ds.PatientBirthDate = "19000101"
            ds.StudyDate = "20000101"
            ds.StudyTime = "120000"
            ds.SeriesInstanceUID, ds.StudyInstanceUID, ds.FrameOfReferenceUID = series, study, frame
            ds.Modality = "MR"
            ds.Manufacturer = "SYNTHETIC"
            ds.MRAcquisitionType = "2D"
            ds.SeriesNumber, ds.InstanceNumber = number, z+1
            ds.SeriesDescription = description
            ds.ImageType = ["ORIGINAL", "PRIMARY", "OTHER"]
            ds.Rows, ds.Columns = 32, 32
            ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [2., 2.], 2., 2.
            ds.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
            ds.ImagePositionPatient = [0., 0., z * 2.]
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
            ds.RepetitionTime, ds.EchoTime, ds.FlipAngle = 2000., 80., 90.
            x, y = np.indices((32, 32))
            array = (1000 * np.exp(-((x-16)**2+(y-16)**2+(z-6)**2)/80)).astype("<u2")
            ds.PixelData = array.tobytes()
            ds.save_as(path, enforce_file_format=True)
    write_json(out / "SYNTHETIC_ONLY.json", {"notice": "Artificial intensity blob. Tests DICOM inventory/conversion only. The DWI placeholder intentionally lacks gradients and must be rejected for reconstruction."})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    print(build(Path(parser.parse_args().output_root)))
