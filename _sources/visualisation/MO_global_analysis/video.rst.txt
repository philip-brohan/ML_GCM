Make the MO global analysis video
=================================

To make the video you need to run the :doc:`plot script <plot>` thousands of times, making an image file for each point in time, and then to merge the thousands of resulting images into a single video.

To make a smooth video we generate one frame each 30 minutes - the temperatures and winds are available on the hour and the precipitation on the half hours, so making frames at 15 and 45 minutes past the hour makes each frame the same (as opposed to having some original data frames and some interpolated data frames).

This script makes a list of commands to plot a frame each 30 minutes over a year:

.. literalinclude:: ../../../visualisation/MO_global_analysis/make_frames.py

You will want to run those jobs in parallel, either with `GNU parallel <https://www.gnu.org/software/parallel/>`_ or by submitting them to a batch system (I used the MO SPICE cluster).

When all the frame images are rendered make a video using `ffmpeg <https://www.ffmpeg.org/>`_. Because of all the detail in the wind and precipitation fields, this video requires a lot of bandwidth, so render it at 20Mbps bandwidth (this is also why the frames are 3840X2160 in size - this produces a 4k video.

.. code-block:: bash

    ffmpeg -r 48 -pattern_type glob -i opfc_global_3var_meanp/\*.png \
           -c:v libx264 -threads 16 -preset veryslow -tune film \
           -profile:v high -level 4.2 -pix_fmt yuv420p -b:v 19M \
           -maxrate 19M -bufsize 20M -c:a copy opfc_global_3var_meanp.mp4
