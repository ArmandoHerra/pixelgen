# Aseprite verification run (PXG-30, research §12)
run_format: 1
date: 2026-10-30T22:23:53Z
operator: orchestrator (OR-20)
aseprite_version: Aseprite 1.3.18.6-dev
os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
python: 3.14.7
pixelgen_version: 0.1.0
pixelgen_commit: eecb0bbedd8cd6da8d5dde562dfc72ef0b7c48e3
display: DISPLAY=unset WAYLAND_DISPLAY=unset
q4_profile_prompt: OWED
cases: pico8-16x16-index0, gameboy-8x8-border, nes-128x128-hashed, endesga32-17x33, grayscale4-16x16-solid
result: PASS
note: 
## Commands
$ $PIXELGEN_ASEPRITE --version
$ $PIXELGEN_ASEPRITE -b pico8-16x16-index0/ours.aseprite --save-as pico8-16x16-index0/aseprite-export.png
$ $PIXELGEN_ASEPRITE -b pico8-16x16-index0/ours.aseprite --script scripts/dump.lua
$ $PIXELGEN_ASEPRITE -b pico8-16x16-index0/ours.aseprite --save-as pico8-16x16-index0/aseprite-resave.aseprite
$ $PIXELGEN_ASEPRITE -b pico8-16x16-index0/aseprite-resave.aseprite --script scripts/dump.lua
$ $PIXELGEN_ASEPRITE -b gameboy-8x8-border/ours.aseprite --save-as gameboy-8x8-border/aseprite-export.png
$ $PIXELGEN_ASEPRITE -b gameboy-8x8-border/ours.aseprite --script scripts/dump.lua
$ $PIXELGEN_ASEPRITE -b gameboy-8x8-border/ours.aseprite --save-as gameboy-8x8-border/aseprite-resave.aseprite
$ $PIXELGEN_ASEPRITE -b gameboy-8x8-border/aseprite-resave.aseprite --script scripts/dump.lua
$ $PIXELGEN_ASEPRITE -b nes-128x128-hashed/ours.aseprite --save-as nes-128x128-hashed/aseprite-export.png
$ $PIXELGEN_ASEPRITE -b nes-128x128-hashed/ours.aseprite --script scripts/dump.lua
$ $PIXELGEN_ASEPRITE -b endesga32-17x33/ours.aseprite --save-as endesga32-17x33/aseprite-export.png
$ $PIXELGEN_ASEPRITE -b endesga32-17x33/ours.aseprite --script scripts/dump.lua
$ $PIXELGEN_ASEPRITE -b endesga32-17x33/ours.aseprite --save-as endesga32-17x33/aseprite-resave.aseprite
$ $PIXELGEN_ASEPRITE -b endesga32-17x33/aseprite-resave.aseprite --script scripts/dump.lua
$ $PIXELGEN_ASEPRITE -b grayscale4-16x16-solid/ours.aseprite --save-as grayscale4-16x16-solid/aseprite-export.png
$ $PIXELGEN_ASEPRITE -b grayscale4-16x16-solid/ours.aseprite --script scripts/dump.lua
## Cases
| case | export_png | dump | resave |
|---|---|---|---|
| pico8-16x16-index0 | PASS | PASS | PASS |
| gameboy-8x8-border | PASS | PASS | PASS |
| nes-128x128-hashed | PASS | PASS | n/a |
| endesga32-17x33 | PASS | PASS | PASS |
| grayscale4-16x16-solid | PASS | PASS | n/a |
## Files
sha256 c991fa5cf0f963658d2430e96f71f06575add02c794f50f934e1c3185352e5fb  endesga32-17x33/aseprite-export.png
sha256 2caac838d9cbdc7624244f459a4523c1d347050d542b159f53218195dee2886d  endesga32-17x33/aseprite-resave.aseprite
sha256 e10eec445639b332d8740cbb507023581c2353953923720f2817399d9c5f182a  endesga32-17x33/dump.txt
sha256 c1ccb227b529d7084809c89f1ff3f91c9f4b6f46ef8c6e7ff3111b666b412148  endesga32-17x33/ours.aseprite
sha256 497d7ff8acc25c9f8bcdae53b8c54ecdae13d1b9b9653edc44ea147da35f9340  gameboy-8x8-border/aseprite-export.png
sha256 b8efe7ceb247251f921fbef9a2a53cb1f9d727d10f3bc52c7a2b0d2a8f030823  gameboy-8x8-border/aseprite-resave.aseprite
sha256 f05c211b9df2e893998dd1b0ecf02757f56b00da5c68fc65bd29d12098f85515  gameboy-8x8-border/dump.txt
sha256 c18d7b392f5488793af387ec10a74c1ad1e21c354193b193fca98ddcda48c23a  gameboy-8x8-border/ours.aseprite
sha256 d08dfab4e92fd657bf4d6b66d1cb924f76297041f76f3511456f783767cf5f4e  grayscale4-16x16-solid/aseprite-export.png
sha256 65465cbd0a8886b17a1e9b1188a18dcd7e57cc0873196297e2228f6dffd45273  grayscale4-16x16-solid/dump.txt
sha256 3d94735b0b4738e44bfc82d6251c256ed8a4edc36dbc4b49303c88053a9ef680  grayscale4-16x16-solid/ours.aseprite
sha256 49b008520cae0635126dea5a84ac1d9efb5af337e32ad23f3b606213f7c16c32  nes-128x128-hashed/aseprite-export.png
sha256 aa01a71f84fa210f2fe781e7f324bd8a6e4b5677895f4676d65f2a9397dd8f95  nes-128x128-hashed/dump.txt
sha256 639766b83703e958ae1d5809a5acdae3f6fe9f5373087ccf448d638d64e55440  nes-128x128-hashed/ours.aseprite
sha256 c341705e82ffff7470c76902df4ddfde5389fd2e31e9b62f11cab3fcb6af4b20  pico8-16x16-index0/aseprite-export.png
sha256 5b38cb96ef77ca8abfde1d89851104ba3295f3093375b9917e4d535548f1d595  pico8-16x16-index0/aseprite-resave.aseprite
sha256 44f47a8079a80e05845edea4e043a95333518fe1cc7d1b4a18657589f34709f9  pico8-16x16-index0/dump.txt
sha256 2ce76771a979b782dce4fb36a256bfb51adde7ddbfbdac8ca326f2335c32571e  pico8-16x16-index0/ours.aseprite
