# CUDA 11.1対応のUbuntuイメージを使用
FROM nvidia/cuda:11.1.1-cudnn8-devel

# 作業ディレクトリの設定
WORKDIR /LT4Code

# apt-get updateの前にこれらを実行し、Nvidiaの公開キーエラーを回避する
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# タイムゾーンを事前に設定しておかないと固まって先に進めない
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-venv python3-pip git gosu && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY LT4Code-main/requirements.txt .

# torchは後からダウンロードし直す
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install allennlp==2.4.0 && \
    pip install transformers==4.5.1

# ホスト名設定スクリプトをコピー    これはホストOSのユーザ名とコンテナ側のユーザ名を一致させるためである
# 開発環境がLinux OSのときのみ有効にする
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# エントリーポイントを設定
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash"]