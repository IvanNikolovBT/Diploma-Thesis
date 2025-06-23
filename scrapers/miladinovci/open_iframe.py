import subprocess
import urllib.parse

def download_pdf_from_viewer_url(viewer_url: str, output_filename: str) -> None:
    
    parsed = urllib.parse.urlparse(viewer_url)
    query_params = urllib.parse.parse_qs(parsed.query)
    encoded_pdf_url = query_params.get('url')
    if not encoded_pdf_url:
        raise ValueError("No 'url' parameter found in viewer URL")
    
    pdf_url = urllib.parse.unquote(encoded_pdf_url[0])
    
    shell_command = f'wget -O {output_filename} "{pdf_url}"'
    
    result = subprocess.run(shell_command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print("Error downloading PDF:", result.stderr)
    else:
        print(f"Downloaded PDF saved as: {output_filename}")

viewer_url = 'https://docs.google.com/viewer?url=https%3A%2F%2Fdigitalna.gbsk.mk%2Ffiles%2Foriginal%2Fdb1c352b624b1d53fef3fcfa2e3c2783.pdf&embedded=true'
output_filename = 'KRAJ.pdf'

download_pdf_from_viewer_url(viewer_url, output_filename)
