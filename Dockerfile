FROM public.ecr.aws/lambda/python:3.12

# instalar dependencia del sistema
RUN dnf install -y libgomp

# copiar código
COPY lambda_function.py .
COPY features.py .
COPY modelo/ ./modelo/

# instalar dependencias
RUN pip install --no-cache-dir pandas numpy lightgbm joblib

# handler
CMD ["lambda_function.lambda_handler"]
