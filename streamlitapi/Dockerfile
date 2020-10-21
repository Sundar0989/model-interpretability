FROM python:3.7-stretch
WORKDIR /streamlit
COPY ./requirements.txt /streamlit
RUN pip install -r requirements.txt
COPY ./webapp.py /streamlit
COPY ./score_objects.pkl /streamlit
EXPOSE 8501
CMD ["streamlit", "run", "webapp.py"]
