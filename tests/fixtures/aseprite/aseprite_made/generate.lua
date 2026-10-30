-- Reference generator: builds a pixelgen-shaped INDEXED sprite and saves it.
-- Usage: aseprite -b --script-param case=<name> --script-param out=<path.aseprite> --script generate.lua
-- NOTE: --script-param flags MUST precede --script.
local case = app.params["case"]
local out = app.params["out"]
if case == nil or out == nil then
  error("usage: --script-param case=<name> --script-param out=<path.aseprite>")
end

local function make_palette(colors)
  -- colors: list of {r,g,b}; slot #colors is the transparent slot.
  local n = #colors
  local pal = Palette(n + 1)
  for i, c in ipairs(colors) do
    pal:setColor(i - 1, Color{ r = c[1], g = c[2], b = c[3], a = 255 })
  end
  pal:setColor(n, Color{ r = 0, g = 0, b = 0, a = 0 })
  return pal
end

local four = { {0,0,0}, {255,0,0}, {0,255,0}, {0,0,255} }
local sixteen = {}
for i = 0, 15 do
  if i == 0 then
    sixteen[#sixteen + 1] = {0,0,0}
  else
    sixteen[#sixteen + 1] = { (i * 16) % 256, (i * 37) % 256, (255 - i * 11) % 256 }
  end
end

local cases = {
  black0 = { w = 8, h = 8, colors = four,
             f = function(x, y) return (x + 2 * y) % 5 end },
  nonsquare = { w = 17, h = 33, colors = four,
                f = function(x, y) return (x * 3 + y) % 5 end },
  grid128 = { w = 128, h = 128, colors = sixteen,
              f = function(x, y) return (x * 7 + y * 13) % 17 end },
  offset_cel = { w = 32, h = 24, colors = four, cel = { x = 5, y = 7, w = 10, h = 6 },
                 f = function(x, y) return (x + y) % 4 end },
}

local spec = cases[case]
if spec == nil then error("unknown case: " .. tostring(case)) end
local n = #spec.colors

local spr = Sprite(spec.w, spec.h, ColorMode.INDEXED)
spr:setPalette(make_palette(spec.colors))
spr.transparentColor = n

-- A fresh Sprite() has one normal (non-background) layer; assert instead of assuming.
local layer = spr.layers[1]
if layer.isBackground then error("default layer is a background layer") end
layer.name = "Layer 1"
layer.isVisible = true

-- Remove the default (empty) cel so we control the stored cel exactly.
for _, cel in ipairs(layer.cels) do spr:deleteCel(cel) end

local cw, ch, cx, cy = spec.w, spec.h, 0, 0
if spec.cel then cw, ch, cx, cy = spec.cel.w, spec.cel.h, spec.cel.x, spec.cel.y end
local img = Image(cw, ch, ColorMode.INDEXED)
for y = 0, ch - 1 do
  for x = 0, cw - 1 do img:drawPixel(x, y, spec.f(x, y)) end
end
spr:newCel(layer, 1, img, Point(cx, cy))

spr:saveAs(out)
