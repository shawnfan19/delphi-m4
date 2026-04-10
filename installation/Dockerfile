FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git curl wget nano vim htop \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Make sure /tmp is writable by everyone (typical for temp dirs)
RUN chmod 1777 /tmp

# Create a general workspace folder
WORKDIR /workspace

# Install requirements
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

# # CRITICAL: Create user and give them ownership of the repo
# # We do this BEFORE switching the WORKDIR so permissions are set correctly
# RUN useradd -m -s /bin/bash delphi_dev \
#     && chown -R delphi_dev:delphi_dev /workspace

# # 6. Switch to the user
# USER delphi_dev
COPY --chmod=755 entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# 7. When you run the container, you start inside /workspace/Delphi
CMD ["/bin/bash"]
