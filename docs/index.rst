Weather Forecasting without the difficult bits
==============================================

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/363005763?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    </table>
    </center>

.. toctree::
   :titlesonly:
   :maxdepth: 1

    Big data: Near-surface temperature, wind, and precipitation, from the Met Office global analysis <visualisation/MO_global_analysis/index>

`Modern weather-forecast models <https://en.wikipedia.org/wiki/Unified_Model>`_ are amazing: amazingly powerful, amazingly accurate - amazingly complicated, amazingly expensive to run and to develop, amazingly difficult to use and to experiment with. Quite often, I'd rather have something less amazing, but much faster and easier to use. Modern `machine learning <https://en.wikipedia.org/wiki/Machine_learning>`_ methods offer sophisticated statistical approximators even to very complex systems like the weather, and we now have hundreds of years of reanalysis output to train them on. How good a model can we build without using any physics, dynamics, chemistry, etc. at all?

My ambition here is to build a `General Circulation Model (GCM) <https://en.wikipedia.org/wiki/General_circulation_model>`_ using *only* machine learning (ML) methods - no hand-specified physics, dynamics, chemistry etc. at all. To that end I'm going to take an existing model, and train a machine learning system to emulate it - the ML version will be less capable as a model of the atmosphere, but should make up for this by being *much* faster, and easier to use. A `previous experiment along these lines <https://github.com/philip-brohan/weather2weather>`_
had `limited success <https://vimeo.com/275778137>`_ but did suggest that good results were possible given the right model architecture.

It would be nice to use the `Met Office UM <https://en.wikipedia.org/wiki/Unified_Model>`_ (see video above) as the model to be emulated, but for a first attempt I'm going to pick an easier target:

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/369615737?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    </table>
    </center>

.. toctree::
   :titlesonly:
   :maxdepth: 1

    Medium-sized data: Near-surface temperature, wind, and mean-sea-level pressure, from the Twentieth Century Reanalysis version 2c <visualisation/20CRv2c_global_analysis/index>

The `Twentieth Century Reanalysis


.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/369615958?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    </table>
    </center>

.. toctree::
   :titlesonly:
   :maxdepth: 1

    Small data: Near-surface temperature, wind, and mean-sea-level pressure, from the Twentieth Century Reanalysis version 2c, after compression into a 100-dimensional latent space. <visualisation/20CRv2c_latent_space/index>


.. toctree::
   :titlesonly:
   :maxdepth: 1

   Download training data <data/download>
   Process training data for TensorFlow<data/conversion>

.. toctree::
   :titlesonly:
   :maxdepth: 1

   A variational autoencoder <models/autoencoder/index>
   A generative model <models/generator/generator>
   A +6hr predictor <models/predictor/index>
   A Machine-Learning GCM using the +6hr predictor <models/GCM/index>
   

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Small Print <credits>

This document and the data associated with it are crown copyright (2019) and licensed under the terms of the `Open Government Licence <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/>`_. All code included is licensed under the terms of the `GNU Lesser General Public License <https://www.gnu.org/licenses/lgpl.html>`_.
