# Dataset access

This code repository contains score labels and fixed group partitions. The original image archive is distributed separately; its review download link will be added here.

After downloading the image archive, extract it before running preprocessing. Set `--image-root` to the extracted directory containing the JPEG files. The image archive is supplied unchanged.

The supplied label table contains 3,360 original filenames and scores. It has not been filtered, rounded or re-split. The column `gade` is the recorded score; `group` identifies the examination group; `fold` is numbered 0-4. The permitted score scale and institutional scoring increment are not inferred from the observed values.

The original records are anonymised mock art-examination records. Use is restricted to non-commercial academic research. The image redistribution route and any additional access terms must follow the data owner's permission. Student identities cannot be linked reliably across groups. The provided partitions establish group disjointness and do not establish student disjointness.
