# generate_cwe_database.py
import requests
import json
import re
import os
from io import BytesIO
import zipfile


def generate_cwe_database():
    """
    Fetch the latest CWE database from MITRE and convert it to a JSON lookup file.
    """
    print("Downloading CWE database from MITRE...")
    url = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Extract the XML from the ZIP
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            xml_filename = z.namelist()[0]  # Usually there's only one file
            with z.open(xml_filename) as f:
                xml_content = f.read().decode("utf-8")

        # Use regex to extract CWE IDs and descriptions (more robust than XML parsing)
        cwe_database = {}

        # Pattern to match Weakness entries
        weakness_pattern = (
            r'<Weakness\s+ID="(\d+)".*?>\s*<Description>(.*?)</Description>'
        )
        matches = re.findall(weakness_pattern, xml_content, re.DOTALL)

        for cwe_id, description in matches:
            # Clean up description (remove XML tags, normalize whitespace)
            clean_desc = re.sub(r"<.*?>", "", description)
            clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

            # Truncate long descriptions
            if len(clean_desc) > 100:
                clean_desc = clean_desc[:97] + "..."

            cwe_database[f"CWE-{cwe_id}"] = clean_desc

        # Pattern to match Category entries
        category_pattern = r'<Category\s+ID="(\d+)".*?>\s*<Name>(.*?)</Name>'
        matches = re.findall(category_pattern, xml_content, re.DOTALL)

        for cat_id, name in matches:
            cwe_database[f"CWE-{cat_id}"] = name.strip()

        # Add common special cases
        cwe_database["NVD-CWE-Other"] = "Weakness not included in CWE list"
        cwe_database["NVD-CWE-noinfo"] = "Insufficient information to assign CWE"

        # Save to JSON file
        with open("cwe_database.json", "w", encoding="utf-8") as f:
            json.dump(cwe_database, f, indent=2, sort_keys=True)

        print(f"Successfully created CWE database with {len(cwe_database)} entries")
        return True

    except Exception as e:
        print(f"Error generating CWE database: {e}")

        # Fall back to creating a basic version with common CWEs
        basic_cwe = {
            "CWE-20": "Improper Input Validation",
            "CWE-22": "Path Traversal",
            "CWE-78": "OS Command Injection",
            "CWE-79": "Cross-site Scripting",
            "CWE-89": "SQL Injection",
            "CWE-94": "Code Injection",
            "CWE-119": "Buffer Overflow",
            "CWE-125": "Out-of-bounds Read",
            "CWE-200": "Information Exposure",
            "CWE-287": "Improper Authentication",
            "CWE-352": "Cross-Site Request Forgery",
            "CWE-400": "Uncontrolled Resource Consumption",
            "CWE-416": "Use After Free",
            "CWE-434": "Unrestricted Upload of File",
            "CWE-502": "Deserialization of Untrusted Data",
            "NVD-CWE-Other": "Weakness not included in CWE list",
            "NVD-CWE-noinfo": "Insufficient information to assign CWE",
        }

        with open("cwe_database.json", "w", encoding="utf-8") as f:
            json.dump(basic_cwe, f, indent=2, sort_keys=True)

        print(f"Created basic CWE database with {len(basic_cwe)} common entries")
        return False


if __name__ == "__main__":
    generate_cwe_database()
