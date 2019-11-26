Make the ML GCM video
=====================

To make the video you need to run the :doc:`plot script <plot>` thousands of times, making an image file for each point in time, and then to merge the thousands of resulting images into a single video.

To make a smooth video we generate one frame every hour. This script makes a list of commands to plot a frame every hour over a year:

.. literalinclude:: ../../../visualisation/GCM_video/make_frames.py

You will want to run those jobs in parallel, either with `GNU parallel <https://www.gnu.org/software/parallel/>`_ or by submitting them to a batch system (I used the MO SPICE cluster).

When all the frame images are rendered make a video using `ffmpeg <https://www.ffmpeg.org/>`_. This video does not need to be at 4k, so render it at 5Mbps bandwidth with frames 1920X1080 in size - standard HD resolution.

.. code-block:: bash

    ffmpeg -r 24 -pattern_type glob -i ML_df_GCM_4var/\*.png \
           -c:v libx264 -threads 16 -preset veryslow -tune film \
           -profile:v high -level 4.2 -pix_fmt yuv420p -b:v 5M \
           -maxrate 5M -bufsize 20M -c:a copy ML_df_GCM_4var.mp4
