FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    curl \
    zsh \
    git \
    vim \
    python3-distutils \
    python3-pip \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install 'alice_satellite @ git+https://github.com/UmbrellaCodr/alice_satellite@main'

RUN chsh -s /bin/zsh root && sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

COPY Docker/zshrc /root/.zshrc
COPY Docker/vimrc /root/.vimrc
COPY Docker/freebsd.vim /root/.vim/styles/freebsd.vim
COPY Docker/samples /root/
COPY Docker/demo.sh /root
COPY tflite /root/alice_data/tflite

# ENTRYPOINT [ "/bin/zsh" ]
# RUN /bin/zsh

CMD ["/bin/zsh","-l"]
