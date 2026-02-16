FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv git curl wget nano vim htop \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Make sure /tmp is writable by everyone (typical for temp dirs)
RUN chmod 1777 /tmp

# 1. Create a general workspace folder
WORKDIR /workspace

# 2. Clone the repo (This creates /workspace/Delphi)
RUN git clone https://github.com/gerstung-lab/Delphi.git

# 3. Install requirements
# Note: We are still in /workspace, so we point to Delphi/requirements.txt
RUN pip install -r Delphi/requirements.txt

# # 4. CRITICAL: Create user and give them ownership of the repo
# # We do this BEFORE switching the WORKDIR so permissions are set correctly
# RUN useradd -m -s /bin/bash delphi_dev \
#     && chown -R delphi_dev:delphi_dev /workspace

# 5. NOW, change the working directory to the repo itself
WORKDIR /workspace/Delphi

# # 6. Switch to the user
# USER delphi_dev

# 7. When you run the container, you start inside /workspace/Delphi
CMD ["/bin/bash"]
