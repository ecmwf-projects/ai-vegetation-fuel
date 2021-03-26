FROM python:3.8.8-slim AS dep-image

# prerequisites
RUN pip install -U pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# User creation
ARG USER_ID
ARG GROUP_ID
ARG USER
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
USER $USER

# install dependencies
COPY requirements.txt .
RUN pip install --user --no-warn-script-location --quiet -r requirements.txt

FROM python:3.8.8-slim AS app-image

# install libgomp1
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# User creation
ARG USER_ID
ARG GROUP_ID
ARG USER
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
USER $USER

# Get python libraries from previous build stage
COPY --from=dep-image /home/$USER/.local /home/$USER/.local

# Make sure scripts in .local are usable:
ENV PATH=/home/$USER/.local/bin:$PATH

# WORKDIR $PROJECT_DIR
RUN mkdir /home/$USER/ml-fuel
WORKDIR /home/$USER/ml-fuel

# --generate-config ensure jupyter config file is created before first launch
RUN jupyter notebook --generate-config
CMD ["jupyter", "notebook", "--port=4242", "--no-browser", "--ip=0.0.0.0", "--allow-root"] 
