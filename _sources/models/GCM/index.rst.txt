A GCM using the 6-hour forecast model
=====================================

The final step to make a GCM is to run the :doc:`predictor <../predictor/index>` repeatedly, using the output from the previous run as the input to the next run. This works, but if the model is left completely free to run it does not have reliable diurnal and annual cycles. I got better results by editing the top-of-atmosphere insolation field in each state (which depends only on time and date) to make sure it was always correct - effectively forcing the model with the TOA radiation pattern.

.. figure:: GCM.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Model structure for the ML GCM (:doc:`Source code <source>`).
   
.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for the ML GCM <source>

To test the Machine Learning GCM, I ran it for a year and made :doc:`a video of the output sequence of atmospheric states. <../../visualisation/GCM_video/index>`

