A 6-hour forecast model based on 20CR
=====================================

.. figure:: predictor.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Model structure for the forecast model (:doc:`Source code <source>`).
   
.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for the forecast model <source>

.. figure:: ../../../models/predictor/compare_tpuv.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Validation of the forecast model. Top panel: T2m, mslp, u and v winds in the original 20CRv3 (at one point in time). Botom panel: same, but after autoencoding. The four scatter-plots compare orignal and encoded values for the four variables. (:doc:`Validation source code <validation_source>`).

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for the forecast model validation <validation_source>
