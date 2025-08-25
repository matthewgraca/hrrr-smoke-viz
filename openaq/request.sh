#!/bin/bash
# LA Extent + PM2.5 sensors + 1000 hits per page
curl --request GET \
  --url "https://api.openaq.org/v3/locations?bbox=-118.75,33.5,-117.0,34.5&parameters_id=2&limit=1000" \
  -s -D headers.txt -o response.json \
  --header "X-API-Key: $OPENAQ_API_KEY"
