A 6-hour forecast model based on 20CR
=====================================

The :doc:`autoencoder generator <../generator/generator>` is a way to make new weather states, but a GCM needs to make a sequence of related new states. A very straighforward way to do this is to re-purpose the :doc:`autoencoder <../autoencoder/index>` to generate the weather 6 hours ahead of the input state, instead of re-generating the input state.

.. figure:: predictor.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Model structure for the forecast model (:doc:`Source code <source>`).
   
This is exactly the same model, and same code, as the :doc:`autoencoder <../autoencoder/index>`, the only change is in the training data: I am using the field at time `t` as the source, and the field at time `t + 6 hours` as the target, where the autoencoder used the field at time `t` for both. (Also, I have reduced the amount of noise added in the regularisation step, as this makes it work better - I don't know why this is.)

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for the forecast model <source>

The validation process for this predictor model is the same as for the :doc:`autoencoder <../autoencoder/index>` - compare generated prediction with the actual field at +6 hours:

.. figure:: ../../../models/predictor/compare_tpuv.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Validation of the forecast model. Top panel: T2m, mslp, u and v winds in the original 20CRv3 (at one point in time). Botom panel: same, but after autoencoding. The four scatter-plots compare orignal and encoded values for the four variables. (:doc:`Validation source code <validation_source>`).

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for the forecast model validation <validation_source>
