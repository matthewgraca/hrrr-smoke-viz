import json
import pandas as pd

# assumes you run the query and stored it in current directory as 'reponse.json'
response_path = 'response.json'
with open(response_path, 'r') as rfile:
    data = json.load(rfile)
    # pretty up the json if we want to manually inspect it
    with open(f'pretty_{response_path}', 'w') as wfile:
        wfile.write(json.dumps(data, indent=4))

    # goal: get sensor IDs, because we use ids to get actual data
    # query for sensors in the extent collecting PM2.5 data
    print(
        "Query used:\n"
        "curl --request GET \\\n"
        "\t--url \"https://api.openaq.org/v3/locations?bbox=-118.75,33.5,-117.0,34.5&parameters_id=2&limit=1000\" \\\n"
        "\t--header \"X-API-Key: $OPENAQ_API_KEY\""
        "\n"
    )
    print(f"Number of sensors in extent: {len(data['results'])}\n")

    # to build the dataframe
    ids = list()
    providers = list()
    locations = list()
    lats, lons = list(), list()

    # further prune list of sensors based on time reporting
    START_DATE = pd.to_datetime("2023-08-02T00:00:00Z")
    END_DATE =  pd.to_datetime("2025-08-02T00:00:00Z")
    provider_ct = dict()
    for res in data['results']:
        name = res['provider']['name']
        start = pd.to_datetime(
            res['datetimeFirst']['utc']  
            if res['datetimeFirst'] is not None
            else END_DATE
        )
        end = pd.to_datetime(
            res['datetimeLast']['utc']
            if res['datetimeLast'] is not None
            else START_DATE
        )
        if start <= START_DATE and end >= END_DATE:
            provider_ct.setdefault(name, 0)
            provider_ct[name] = provider_ct[name] + 1
            ids.append(res['id'])
            providers.append(name)
            locations.append(res['name'])
            lats.append(res['coordinates']['latitude'])
            lons.append(res['coordinates']['longitude'])

    pd.set_option("display.max_rows", None)
    df = pd.DataFrame({
        'id' : ids,
        'provider' : providers,
        'locations' : locations,
        'latitude' : lats,
        'longitude' : lons
    })

    print(
        "Providers that meet the criteria:\n"
        "\t1. Within extent\n"
        "\t2. Within date range\n"
        "\t3. Tracks PM2.5\n"
        f"Number of sensors that meet criteria: {len(ids)}\n"
        f"Count: {provider_ct}\n"
        f"Ids: {ids[:5]} ... {ids[-5:]}"
    )

    print(df)

    # once you have the sensor ids, you can query? Seems like it'd hit the api limit fast, with 200+ sensors.
