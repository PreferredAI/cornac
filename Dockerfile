###########
# BUILDER #
###########

FROM python:3.11.6-slim-bullseye AS builder

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./setup.py setup.py
COPY ./cornac cornac
COPY ./README.md README.md

RUN pip install --upgrade pip
RUN pip install Cython numpy scipy

RUN apt-get update && \
    apt-get -y --no-install-recommends install gcc g++

RUN pip install --no-cache-dir . # install cornac

##########
# RUNNER #
##########

FROM python:3.11.6-slim-bullseye AS runner

WORKDIR /app

# Set environment variables
ENV MODEL_PATH=""
ENV MODEL_CLASS=""
ENV PORT=5000

COPY --from=builder /app/cornac cornac
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN apt-get update && \
    apt-get -y --no-install-recommends install gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir Flask gunicorn

WORKDIR /app/cornac/serving

CMD ["gunicorn", "app:app"]
