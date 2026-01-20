FROM alphaunito/streamflow:0.2.0

WORKDIR /opt/qsplit
COPY . /opt/qsplit

# Installa QSplit (e quindi il plugin entry-point) dentro l’ambiente Python del container
RUN pip install --no-cache-dir -e .[streamflow,ibm-quantum,dwave,iqm]