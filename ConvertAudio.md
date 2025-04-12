### macOS / linux

# Navigate to your project directory
cd your_project_directory

# Create WAV versions of all AIF files
mkdir -p assets/sounds_wav
for file in assets/sounds/*.aif; do
    filename=$(basename -- "$file")
    name="${filename%.*}"
    ffmpeg -i "$file" "assets/sounds_wav/${name}.wav"
done

# Optionally, replace original files
# mv assets/sounds_wav/* assets/sounds/


### Windows

# Navigate to your project directory
cd your_project_directory

# Create WAV versions of all AIF files
mkdir -Force assets/sounds_wav
Get-ChildItem -Path "assets/sounds" -Filter "*.aif" | ForEach-Object {
    $outputName = "assets/sounds_wav/" + $_.BaseName + ".wav"
    ffmpeg -i $_.FullName $outputName
}

# Optionally, replace original files
# Move-Item -Path assets/sounds_wav/* -Destination assets/sounds/ -Force