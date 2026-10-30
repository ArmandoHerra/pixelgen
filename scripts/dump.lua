-- Prints the PXG-30 dump format v1 for the active sprite (read by verify_aseprite.parse_dump).
-- Usage: aseprite -b FILE.aseprite --script scripts/dump.lua
-- Rows are the frame-1 composite as palette indices: a W x H canvas pre-filled with the
-- transparent index, layers drawn bottom to top at their cel position, and a pixel equal to the
-- transparent index never overwrites.
local spr = app.activeSprite
if spr == nil then
  error("dump.lua: no active sprite")
end

local mode = "other"
if spr.colorMode == ColorMode.INDEXED then
  mode = "indexed"
elseif spr.colorMode == ColorMode.RGB then
  mode = "rgb"
elseif spr.colorMode == ColorMode.GRAY then
  mode = "grayscale"
end

local transparent = spr.transparentColor
local pal = spr.palettes[1]

local canvas = {}
for y = 0, spr.height - 1 do
  local row = {}
  for x = 0, spr.width - 1 do
    row[x] = transparent
  end
  canvas[y] = row
end

local function draw(layers)
  for _, layer in ipairs(layers) do
    if layer.isGroup then
      if layer.isVisible then
        draw(layer.layers)
      end
    elseif layer.isVisible then
      local cel = layer:cel(1)
      if cel ~= nil then
        local img = cel.image
        for v = 0, img.height - 1 do
          local cy = cel.position.y + v
          if cy >= 0 and cy < spr.height then
            for u = 0, img.width - 1 do
              local cx = cel.position.x + u
              if cx >= 0 and cx < spr.width then
                local value = img:getPixel(u, v)
                if value ~= transparent then
                  canvas[cy][cx] = value
                end
              end
            end
          end
        end
      end
    end
  end
end
draw(spr.layers)

print("PXGDUMP 1")
print("size " .. spr.width .. " " .. spr.height)
print("mode " .. mode)
print("transparent " .. transparent)
print("ncolors " .. #pal)
for i = 0, #pal - 1 do
  local c = pal:getColor(i)
  print("palette " .. i .. " " .. c.red .. " " .. c.green .. " " .. c.blue .. " " .. c.alpha)
end
for y = 0, spr.height - 1 do
  local vals = {}
  for x = 0, spr.width - 1 do
    vals[#vals + 1] = tostring(canvas[y][x])
  end
  print("row " .. table.concat(vals, " "))
end
print("end")
