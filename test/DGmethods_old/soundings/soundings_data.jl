
"""
    extract_targz(file)

Platform-independent file extraction
"""
function extract_targz(file)
  if Sys.iswindows()
    run(pipeline(`7z x -tgzip -so $file`, `7z x -si -ttar`))
  else
    run(`tar -xzf $file`)
  end
end

function soundings_data_folder()
  register(DataDep("SoundingsData",
                   "Soundings data",
                   "https://caltech.box.com/shared/static/lkwllh1rayxqole207ccq70scxhhjvzt.gz",
                   "7b97458cca8647f5190e6e09961d5056068843d07542d32bc5ea03d9f15ced94",
                   post_fetch_method=extract_targz))
  datafolder = datadep"SoundingsData"
  return datafolder
end
