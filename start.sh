#!/bin/bash

uvicorn ollm_api.server:app --host 0.0.0.0 $@