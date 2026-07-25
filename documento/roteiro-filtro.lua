
local MOLDE = "output/slides/slide-%02d.png"

function HorizontalRule()
  return {}
end

local function numero_no_deck(titulo)
  return tonumber(titulo:match("slide%s+(%d+)%s+do%s+deck"))
      or tonumber(titulo:match("^Slide%s+(%d+)"))
end

local function miniatura(titulo)
  local n = numero_no_deck(titulo)
  if not n then return nil end
  local arquivo = string.format(MOLDE, n)
  local f = io.open(arquivo, "r")
  if not f then
    io.stderr:write("aviso: miniatura ausente para " .. titulo .. " (" .. arquivo .. ")\n")
    return nil
  end
  f:close()
  return pandoc.RawBlock("latex", "\\miniaturaslide{" .. arquivo .. "}")
end

function Pandoc(doc)
  local blocos, comecou = {}, false
  for _, bloco in ipairs(doc.blocks) do
    local titulo = nil
    if bloco.t == "Header" and bloco.level == 2 then
      titulo = pandoc.utils.stringify(bloco)
      if titulo:match("^Slide") then comecou = true end
    end
    if comecou then
      table.insert(blocos, bloco)
      if titulo then
        local guia = miniatura(titulo)
        if guia then table.insert(blocos, guia) end
      end
    end
  end
  doc.blocks = blocos
  return doc
end
