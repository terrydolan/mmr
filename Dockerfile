# mapmyrun data analysis web app
FROM python:3.7-slim
LABEL MAINTAINER=terry8dolan@gmail.com
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run", "mapmyrun_app.py" ]