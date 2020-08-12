FROM continuumio/miniconda3:4.8.2

RUN groupadd --gid 1000 worker && useradd --uid 1000 --gid worker -ms /bin/bash worker
WORKDIR /app
RUN chown -R worker:worker /app

USER worker

# Create app directory

# Install app dependencies
COPY --chown=worker:worker ./ /app
ENV PATH="/home/worker/.local/bin:${PATH}"

RUN conda env create -f /app/environment.yml
RUN sed -i '$ d' ~/.bashrc && \
    echo "conda activate torch" >> ~/.bashrc

RUN python -m spacy download en_core_web_lg
# Bundle app source


ENV PYTHONPATH="/app:${PYTHONPATH}"
CMD [ "python", "webapp/api.py"]