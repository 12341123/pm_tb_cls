import numpy as np
import math
from struct import unpack

def convert_dicom_v2(ds, *, index=0, output_bits=8, full_range=False,
                     prefer_lut=False, force_monochrome2=True):
    """Converts DICOM into displayable image.

    Args:
        ds (pydicom.dataset.FileDataset): DICOM FileDataset object.
        index (int):
        output_bits (int):
        full_range (bool):
        prefer_lut (bool):
        force_monochrome2 (bool):
    """
    arr = ds.pixel_array
    original_dtype = arr.dtype

    valid_lut = False
    if "VOILUTSequence" in ds:
        valid_lut = None not in [
            ds.VOILUTSequence[0].get(n, None)
            for n in ("LUTDescriptor", "LUTData")
        ]
    valid_windowing = None not in [
        ds.get(n, None) for n in ("WindowCenter", "WindowWidth")
    ]

    if (not prefer_lut or not valid_lut) and valid_windowing:
        if ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']:
            raise ValueError(
                "When performing a windowing operation only 'MONOCHROME1' and "
                "'MONOCHROME2' are allowed for (0028,0004) Photometric "
                "Interpretation"
            )

        # VR DS, VM 1-n
        elem = ds["WindowCenter"]
        center = elem.value[index] if elem.VM > 1 else elem.value
        elem = ds["WindowWidth"]
        width = elem.value[index] if elem.VM > 1 else elem.value

        # The output range depends on whether or not a modality LUT or rescale
        # operation has been applied
        if 'ModalityLUTSequence' in ds:
            # Unsigned - see PS3.3 C.11.1.1.1
            y_min = 0
            try:
                bit_depth = ds.ModalityLUTSequence[0].LUTDescriptor[2]
                y_max = 2**bit_depth - 1
            except:
                y_min = -2 ** (ds.BitsStored - 1)
                y_max = 2 ** (ds.BitsStored - 1) - 1
        elif ds.PixelRepresentation == 0:  # Unsigned
            y_min = 0
            y_max = 2**ds.BitsStored - 1
        else:  # Signed
            y_min = -2**(ds.BitsStored - 1)
            y_max = 2**(ds.BitsStored - 1) - 1

        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
            # Otherwise its the actual data range
            y_min = y_min * ds.RescaleSlope + ds.RescaleIntercept
            y_max = y_max * ds.RescaleSlope + ds.RescaleIntercept

        y_range = y_max - y_min
        arr = arr.astype('float64')

        # May be LINEAR (default), LINEAR_EXACT, SIGMOID or not present, VM 1
        voi_func = getattr(ds, "VOILUTFunction", "LINEAR").upper()
        if voi_func in ['LINEAR', 'LINEAR_EXACT']:
            # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
            if voi_func == 'LINEAR':
                if width < 1:
                    raise ValueError(
                        "The (0028,1051) Window Width must be greater than or "
                        "equal to 1 for a 'LINEAR' windowing operation"
                    )
                center -= 0.5
                width -= 1
            elif width <= 0:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than 0 "
                    "for a 'LINEAR_EXACT' windowing operation"
                )

            below = arr <= (center - width / 2)
            above = arr > (center + width / 2)
            between = np.logical_and(~below, ~above)

            arr[below] = y_min
            arr[above] = y_max
            if between.any():
                arr[between] = (
                        ((arr[between] - center) / width + 0.5) * y_range + y_min
                )
        elif voi_func == 'SIGMOID':
            # PS3.3 C.11.2.1.3.1
            if width <= 0:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than 0 "
                    "for a 'SIGMOID' windowing operation"
                )

            arr = y_range / (1 + np.exp(-4 * (arr - center) / width)) + y_min
        else:
            raise ValueError(
                "Unsupported (0028,1056) VOI LUT Function value '{}'"
                .format(voi_func)
            )

        if force_monochrome2 and ds.PhotometricInterpretation == "MONOCHROME1":
            arr = y_min + y_max - arr
    elif valid_lut:
        # VOI LUT Sequence contains one or more items
        item = ds.VOILUTSequence[index]
        nr_entries = item.LUTDescriptor[0] or 2**16
        first_map = item.LUTDescriptor[1]

        # PS3.3 C.8.11.3.1.5: may be 8, 10-16
        nominal_depth = item.LUTDescriptor[2]
        if nominal_depth in list(range(10, 17)):
            dtype = 'uint16'
        elif nominal_depth == 8:
            dtype = 'uint8'
        else:
            raise NotImplementedError(
                "'{}' bits per LUT entry is not supported"
                .format(nominal_depth)
            )

        # Ambiguous VR, US or OW
        if item['LUTData'].VR == 'OW':
            endianness = '<' if ds.is_little_endian else '>'
            unpack_fmt = '{}{}H'.format(endianness, nr_entries)
            lut_data = unpack(unpack_fmt, item.LUTData)
        else:
            lut_data = item.LUTData
        lut_data = np.asarray(lut_data, dtype=dtype)

        # IVs < `first_map` get set to first LUT entry (i.e. index 0)
        clipped_iv = np.zeros(arr.shape, dtype=arr.dtype)
        # IVs >= `first_map` are mapped by the VOI LUT
        # `first_map` may be negative, positive or 0
        mapped_pixels = arr >= first_map
        clipped_iv[mapped_pixels] = arr[mapped_pixels] - first_map
        # IVs > number of entries get set to last entry
        np.clip(clipped_iv, 0, nr_entries - 1, out=clipped_iv)
        arr = lut_data[clipped_iv]

        y_min, y_max = 0, 2 ** nominal_depth - 1
    else:
        if "BitsStored" in ds:
            y_min, y_max = 0, 2 ** ds.BitsStored - 1
            arr = arr & y_max  # FIXME: is this the right way to clip pixel array?
            if force_monochrome2 and ds.PhotometricInterpretation == "MONOCHROME1":
                arr = y_min + y_max - arr
        else:
            raise ValueError("No BitsStored found in this DICOM!")

    if output_bits is None:
        return arr.astype(original_dtype)
    else:
        if full_range:
            y_min, y_max = np.amin(arr), np.amax(arr)
        y_range = y_max - y_min
        arr = (arr.astype("float64") - y_min) / y_range * (2 ** output_bits - 1)
        dtype = f"uint{math.ceil(output_bits / 8) * 8}"
        return arr.astype(dtype)
