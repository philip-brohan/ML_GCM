A variational autoencoder for 20CR
==================================

.. figure:: autoencoder.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Model structure for the variational autoencoder (:doc:`Source code <source>`).

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for the variational autoencoder <source>

.. figure:: ../../../models/autoencoder/compare_tpuv.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Validation of the autoencoder. Top panel: T2m, mslp, u and v winds in the original 20CRv3 (at one point in time). Botom panel: same, but after autoencoding. The four scatter-plots compare orignal and encoded values for the four variables. (:doc:`Validation source code <validation_source>`).

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for the autoencoder validation <validation_source>

