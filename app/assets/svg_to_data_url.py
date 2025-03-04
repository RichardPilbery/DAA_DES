import urllib.parse

def svg_to_data_url(svg_path):
    with open(svg_path, "r", encoding="utf-8") as file:
        svg_content = file.read()
    encoded_svg = urllib.parse.quote(svg_content)
    return f"data:image/svg+xml,{encoded_svg}"

print(svg_to_data_url("app/assets/daa-logo_scoured.svg"))
