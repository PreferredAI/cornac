###########
# BUILDER #
###########

FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./setup.py setup.py
COPY ./cornac cornac
COPY ./README.md README.md

RUN apt-get update && \
    apt-get -y --no-install-recommends install gcc g++ && \
    pip install --no-cache-dir Cython numpy scipy && \
    pip install --no-cache-dir . 

# RUN pip install --no-cache-dir cornac # install cornac

##########
# RUNNER #
##########

FROM python:3.11-slim AS runner

WORKDIR /app

# Set environment variables
ENV MODEL_PATH=""
ENV MODEL_CLASS=""
ENV PORT=5000

COPY --from=builder /app/cornac/serving cornac/serving
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN apt-get update && \
    apt-get -y --no-install-recommends install libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir Flask gunicorn

WORKDIR /app/cornac/serving

CMD ["gunicorn", "app:app"]
