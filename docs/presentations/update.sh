#!/bin/bash

./sync.py --bucket=philip.brohan.org.big-files --prefix=ML_GCM --name=ML_GCM_poster.key

./sync.py --bucket=philip.brohan.org.big-files --prefix=ML_GCM --name=ML_GCM_poster.pdf

./sync.py --bucket=philip.brohan.org.big-files --prefix=ML_GCM --name=MetO_ML_20191126.key

./sync.py --bucket=philip.brohan.org.big-files --prefix=ML_GCM --name=MetO_ML_20191126.pptx

convert -geometry 1080x764 ML_GCM_poster.pdf ML_GCM_poster.png
