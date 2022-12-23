FROM continuumio/miniconda3
COPY adv_env.txt adv_env.txt
RUN conda create --name adv_env --file adv_env.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install accelerate
RUN pip install transformers 
RUN pip install diffusers
RUN pip install nltk
RUN pip install gpytorch 
RUN pip install botorch
RUN pip install wandb
RUN pip install fire 
RUN pip install matplotlib
RUN pip install pandas
ENV WANDB_API_KEY=dfa956c5bfb9fa492ebf9adede99093772919518

# docker build -t nmaus/advenv .
# dockerd-rootless-setuptool.sh install
# systemctl --user start docker
# export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
# docker run -v /home1/n/nmaus/adversarial_img_bo/:/workspace/ --gpus all -it nmaus/advenv
