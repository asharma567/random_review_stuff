#!/bin/bash -xe



# Non-standard and non-Amazon Machine Image Python modules:
sudo pip install -U \
  awscli            \
  boto              \
  ciso8601          \
  ujson             \
  workalendar       \
  pandas            \
  py-stringmatching
sudo yum install -y python-psycopg2