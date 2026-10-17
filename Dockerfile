# Pre-cached GPU ML Base Image with NVIDIA CUDA + XGBoost 3.2 GPU + LightGBM + Scikit-Learn + Starlette/Uvicorn
FROM freqtradeorg/freqtrade:develop_freqai

USER root
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install remaining lightweight HTML scraping & visualization libraries
RUN pip install --no-cache-dir beautifulsoup4 matplotlib seaborn

# Include pre-installed GPU ML & Web packages from ftuser site-packages
ENV PYTHONPATH="/home/ftuser/.local/lib/python3.14/site-packages"
ENV PATH="/home/ftuser/.local/bin:${PATH}"

# Copy project source and data
COPY data/ /app/data/
COPY outputs/ /app/outputs/
COPY src/ /app/src/

EXPOSE 8000

ENTRYPOINT ["python"]
CMD ["src/app.py", "--host", "0.0.0.0", "--port", "8000"]
