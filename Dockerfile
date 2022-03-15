FROM Python:3.9.0
ADD . /code
WORKDIR /code
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
CMD python main.py
