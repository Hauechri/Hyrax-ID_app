# How the app finds your recordings

You point the app at two things: an **Annotation file** and an **Audio folder**.
There is exactly **one rule** connecting them:

> The **"Song ID"** column in the Annotation file must equal the **filename**
> of a `.wav` file in the Audio folder (without the `.wav` part).

That's it. Animal names and folder names are **never** compared to anything —
they don't need to match each other at all. The folder can be organised
however you like.

---

## 1. Annotation file (the Excel file)

One row per bout. It needs these columns (same names as the file you have
now):

| Singer | Song ID | Bout number | Time start | Time end | Hzmin | Hzmax | Element |
|--------|---------|-------------|------------|----------|-------|-------|---------|
| MS | 240701_0091 | 1 | 26.09 | 27.14 | 1400 | 2710 | Wail |

- **Singer** = the animal's name. This is only used to build the "Animal"
  dropdown in the app — it is not checked against any folder name.
- **Song ID** = the recording's name. **This is the one value that must
  match a real audio file** (see below).

## 2. Audio folder (the .wav files)

Any layout works:
- all `.wav` files loose in one folder, **or**
- one subfolder per animal, **or**
- nested however you like, at any depth.

The app searches the whole folder for `.wav` files and matches each one to
the Annotation file purely by name.

**Example:** if the Annotation file has a row with `Song ID = 240701_0091`,
the app looks for a file named `240701_0091.wav` *anywhere* inside the
Audio folder you picked — it doesn't matter if that's:

```
INPUT/240701_0091.wav
INPUT/MS/240701_0091.wav
INPUT/anything/you/want/240701_0091.wav
```

All three work identically.

---

## What goes wrong (and what the app tells you)

- **An animal has no recordings in the dropdown** → none of its Song IDs
  have a matching `.wav` file anywhere in the folder you picked.
- **Nothing loads at all** → you'll get a popup listing the animal names
  found in the Excel file, as a hint that the audio folder doesn't contain
  any matching filenames yet.
- **Two different animals accidentally share a Song ID / filename** → the
  app will silently use whichever one it finds first. This shouldn't happen
  with the date-based recording names in this dataset, but it's the one
  case the app can't warn you about.
