FROM conda/miniconda2

RUN mkdir /ai-workshop \
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y build-essential 

WORKDIR /ai-workshop

ADD data/ data/  
ADD day1/ day1/
ADD day2/ day2/ 
ADD environment.yml .

RUN conda env create -f environment.yml

EXPOSE 8888