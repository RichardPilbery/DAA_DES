function detect_icon_type(icon)
  if icon:match("%.svg$") then
    return "svg"
  elseif icon:match("%.png$") or icon:match("%.jpe?g$") or icon:match("%.webp$") then
    return "png"
  elseif icon:match("^fa[srbldt]?%-.+") then
    return "fa"
  elseif icon:match("^ti%-.+") then
    return "tabler"
  elseif icon:match("^ph%-?%a*%s+ph%-.+") then
    return "phosphor"
  else
    return "bi"  -- fallback: assume Bootstrap Icons
  end
end

-- Material Symbols variants: never auto-detected (icon names are plain words
-- like "home", indistinguishable from a bare Bootstrap Icons fallback), so
-- these are only reachable via an explicit icon-type attribute.
local material_variants = {
  ["material"]          = { class = "material-symbols-outlined", family = "Material+Symbols+Outlined" },
  ["material-outlined"] = { class = "material-symbols-outlined", family = "Material+Symbols+Outlined" },
  ["material-rounded"]  = { class = "material-symbols-rounded",  family = "Material+Symbols+Rounded" },
  ["material-sharp"]    = { class = "material-symbols-sharp",    family = "Material+Symbols+Sharp" },
}

-- Phosphor Icons ships one stylesheet per weight rather than a single bundle,
-- so the leading weight class in the icon value (e.g. "ph-bold ph-heart")
-- picks which CDN file gets linked.
local phosphor_weight_dirs = {
  ["ph"]         = "regular",
  ["ph-thin"]    = "thin",
  ["ph-light"]   = "light",
  ["ph-bold"]    = "bold",
  ["ph-fill"]    = "fill",
  ["ph-duotone"] = "duotone",
}

-- Icon webfont stylesheets. The two *_FMT entries are string.format templates,
-- not usable URLs: the Material Symbols font family and the Phosphor weight
-- directory are substituted in at the point of use.
local BOOTSTRAP_ICONS_CSS      = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
local FONT_AWESOME_CSS         = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
local TABLER_ICONS_CSS         = "https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.46.0/dist/tabler-icons.min.css"
local MATERIAL_SYMBOLS_CSS_FMT = "https://fonts.googleapis.com/css2?family=%s:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block"
local PHOSPHOR_ICONS_CSS_FMT   = "https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.2/src/%s/style.css"

-- Link an external icon stylesheet into the document head.
--
-- Registered as an HTML dependency rather than via quarto.doc.include_text for
-- two reasons. Quarto deduplicates dependencies, so a document with many boxes
-- gets one <link> per stylesheet instead of one per box; and HTML dependencies
-- are only consumed when rendering to HTML, whereas include_text("in-header")
-- injects into every format's header — which drops a raw <link> tag into the
-- LaTeX preamble on PDF renders.
--
-- The dependency name is derived from the href because Quarto dedupes on name
-- alone and ignores version. Stylesheets differing only in the substituted
-- family or weight (e.g. two Material Symbols variants) must therefore produce
-- distinct names, so a shared hand-written name would silently drop every
-- variant after the first.
local function include_icon_stylesheet(href)
  quarto.doc.add_html_dependency({
    name = "value-box-icons-" .. href:gsub("[^%w]", "-"),
    version = "1.0.0",
    links = {{ rel = "stylesheet", href = href }}
  })
end

-- Link the extension's own stylesheet into the document head. Both Div
-- branches (value-box and value-box-row) need this, so it's a shared
-- one-line call rather than two independently-maintained copies of the same
-- dependency table — Quarto dedupes by name regardless of how many times
-- either branch calls this.
local function include_value_box_stylesheet()
  quarto.doc.add_html_dependency({
    name = "value-box-styles",
    version = "1.0.0",
    stylesheets = {"value-box.css"}
  })
end

-- Build a CSS declaration, or nothing at all when the value is empty.
-- Attributes that default to "" (height, font-size) would otherwise emit
-- "height:;", an invalid declaration that browsers discard silently. Also
-- used to build custom-property declarations (e.g. property="--vb-width") —
-- the property/value split is the same either way.
local function css_decl(property, value)
  if value == nil or value == "" then
    return ""
  end
  return string.format("%s:%s; ", property, value)
end

-- Reads width/height straight out of a PNG's IHDR chunk (bytes 16-23,
-- after the 8-byte signature and the chunk's own 4-byte length + 4-byte
-- "IHDR" type) rather than pulling in an image library just to learn the
-- icon's aspect ratio. Returns nil, nil if the file is too short or isn't
-- actually a PNG.
local function png_dimensions(path)
  local f = io.open(path, "rb")
  if not f then return nil, nil end
  local header = f:read(24)
  f:close()
  if not header or #header < 24 or header:sub(1, 8) ~= "\137PNG\r\n\26\n" then
    return nil, nil
  end
  local function be32(offset)
    local b1, b2, b3, b4 = header:byte(offset, offset + 3)
    return b1 * 16777216 + b2 * 65536 + b3 * 256 + b4
  end
  return be32(17), be32(21)
end

-- Reads the intrinsic aspect ratio out of an already-loaded SVG file's own
-- viewBox (falling back to its width/height attributes if it has no
-- viewBox). Only the first match is read, matching the assumption the
-- style-injection gsub elsewhere already makes: that the file's own root
-- <svg ...> tag is the first one in the document.
local function svg_dimensions(svg_content)
  local _minx, _miny, w, h = svg_content:match(
    'viewBox%s*=%s*"%s*([%-%d%.]+)[%s,]+([%-%d%.]+)[%s,]+([%-%d%.]+)[%s,]+([%-%d%.]+)')
  if w and h then
    return tonumber(w), tonumber(h)
  end
  local aw = svg_content:match('<svg[^>]-%swidth%s*=%s*"([%d%.]+)')
  local ah = svg_content:match('<svg[^>]-%sheight%s*=%s*"([%d%.]+)')
  if aw and ah then
    return tonumber(aw), tonumber(ah)
  end
  return nil, nil
end

-- Chooses which axis icon-size constrains: whichever is larger for the
-- icon's own natural dimensions, with the other left auto so the browser
-- scales it proportionally. Without this, forcing both width and height to
-- the same value pins a non-square icon into a square box and it gets
-- letterboxed inside it — visible as empty space the icon's own background
-- shows through. Falls back to the old fixed square when the natural
-- dimensions can't be determined (unexpected format, corrupt file, no
-- viewBox); the second return value flags that fallback so callers that
-- also use object-fit:contain as a safety net know when they still need it.
local function icon_size_style(size, natural_w, natural_h)
  if not natural_w or not natural_h or natural_w <= 0 or natural_h <= 0 then
    return css_decl("width", size) .. css_decl("height", size), true
  elseif natural_w >= natural_h then
    return css_decl("width", size) .. "height:auto; ", false
  else
    return "width:auto; " .. css_decl("height", size), false
  end
end

-- Escape a value for interpolation into a double-quoted HTML attribute.
local function escape_attr(s)
  return (tostring(s):gsub('[&<>"]', {
    ["&"] = "&amp;",
    ["<"] = "&lt;",
    [">"] = "&gt;",
    ['"'] = "&quot;",
  }))
end

-- Build a style declaration for a *themeable default*: emit it only when the
-- box (or its .value-box-row, via inheritance) actually set the attribute.
-- When it didn't, emit nothing — value-box.css's own `var(--vb-*, <default>)`
-- fallback then supplies the default, which is also what lets a project
-- retarget that default on :root. A per-box/per-row attribute still wins,
-- because it lands in the element's inline style. `raw` is the unescaped
-- attribute value straight from el.attributes (Lua nil when unset; "" — which
-- is truthy in Lua — counts as unset too, matching icon-size's handling).
local function themed_decl(raw, property)
  if raw == nil or raw == "" then
    return ""
  end
  return css_decl(property, escape_attr(raw))
end

-- Shared by value-box and value-box-row: pass through the div's own #id, any
-- classes beyond own_class, and the data-*/aria-*/role/tabindex/lang
-- namespace, exactly as an ordinary Pandoc div would. A literal style="..."
-- is dropped (with a warning naming extra_style_hint as the escape hatch to
-- use instead) rather than colliding with the wrapper's own generated style.
-- reserved_attr, when given, additionally drops one further attribute name
-- that the caller generates itself (value-box's data-fragment-index) so a
-- literal copy can't collide with it — {name=..., active=..., warning=...}.
--
-- Factored out rather than duplicated per caller: this repo has already
-- shipped one real regression from this exact logic (the reserved-namespace
-- matching had a case-sensitivity bug — see the v1.4.0 changelog entry), so
-- two independently-maintained copies would be a real drift hazard, not just
-- repetition.
local function build_wrapper_attrs(el, own_class, extra_style_hint, reserved_attr)
  local id_attr = ""
  if el.identifier ~= "" then
    id_attr = string.format(' id="%s"', escape_attr(el.identifier))
  end

  local extra_classes = ""
  for _, class in ipairs(el.classes) do
    if class ~= own_class then
      extra_classes = extra_classes .. " " .. escape_attr(class)
    end
  end

  -- Attributes that pass through onto the wrapper: the two namespaces built
  -- for exactly this (data-*, aria-*), plus role/tabindex/lang — the standard
  -- globals most likely to actually matter (ARIA landmarks/labelling,
  -- keyboard reachability, screen-reader pronunciation). Anything else is not
  -- part of this filter's own vocabulary and is not passed through.
  --
  -- This is an allowlist rather than a denylist deliberately — see the
  -- CONTRIBUTING.md note on why (a denylist meant every new filter option had
  -- to be added to a second list too, and silently renamed something like
  -- onclick="..." into an inert data-onclick).
  local passthrough_names = {
    ["role"] = true, ["tabindex"] = true, ["lang"] = true,
  }
  -- Keyed on the lowercased attribute name throughout — HTML attribute names
  -- are case-insensitive, and Quarto's own HTML postprocessing lowercases
  -- them regardless, so a mixed-case key like "Style" or "Data-Id" must still
  -- be recognised as the thing it is.
  local passthrough_attrs = ""
  for key, val in pairs(el.attributes) do
    local key_lower = key:lower()
    if key_lower == "style" then
      -- A literal style="..." would collide with the style attribute this
      -- filter builds for the same element. Two style attributes on one tag
      -- isn't undefined — HTML parsers keep the first and silently drop the
      -- second — so without this, a user's own style would be dropped with
      -- no warning at all.
      io.stderr:write(string.format(
        "value-box warning: a literal 'style' attribute is ignored to avoid clashing with the %s's own style — use %s instead\n",
        own_class, extra_style_hint))
    elseif reserved_attr and key_lower == reserved_attr.name and reserved_attr.active then
      io.stderr:write(reserved_attr.warning)
    elseif passthrough_names[key_lower] or key_lower:match("^data%-") or key_lower:match("^aria%-") then
      passthrough_attrs = passthrough_attrs .. string.format(' %s="%s"', key_lower, escape_attr(val))
    end
  end

  return id_attr, extra_classes, passthrough_attrs
end

-- Attributes eligible to flow down from .value-box-row to a .value-box child
-- that doesn't set them itself (a child's own value, including an explicit
-- blank override, always wins — see inject_row_defaults below). Deliberately
-- an explicit allowlist rather than "everything except a denylist", matching
-- build_wrapper_attrs's own passthrough-namespace convention above.
--
-- Excluded on purpose: icon/value/title/delta/delta-direction/href (per-box
-- content/identity, the whole reason a row has more than one box), index
-- (must stay unique per box for Reveal.js fragment ordering), fragment
-- (per-box animation opt-in), and icon-type — despite looking like a styling
-- switch, it's the only way to opt into Material Symbols (never
-- auto-detected, see the comment above material_variants), so inheriting it
-- would silently coerce every other child's icon (e.g. an "fa-star" box)
-- onto the Material renderer too, with no warning. Each box sets its own
-- icon-type when it needs one.
local INHERITABLE_ATTRS = {
  "icon-position", "icon-size", "icon-color",
  "color",
  "width", "height", "min-height", "padding",
  "align", "valign",
  "font-size", "font-color",
  "value-position", "value-font-size", "value-color",
  "title-font-size", "title-color",
  "delta-color", "delta-font-size",
  "target",
  "outer-extra-style", "icon-extra-style", "content-extra-style",
  "details-extra-style", "value-extra-style", "title-extra-style",
  "delta-extra-style", "value-row-extra-style",
}

-- Runs as its own filter pass, before the Div(el) pass below (see the
-- `return` at the end of this file). Pandoc's single-pass Div(el) walk is
-- bottom-up, so by the time it reaches a .value-box-row its .value-box
-- children have already been expanded into raw HTML — nothing left to read
-- attributes off. Running this as an earlier, separate pass mutates each
-- child's attributes table in place while it's still a real Div, so the
-- later Div(el) pass picks up the merged values through its existing
-- `el.attributes["x"] or default` reads with no changes to that logic at all.
local function inject_row_defaults(el)
  if not el.classes:includes("value-box-row") then
    return nil
  end
  for _, child in ipairs(el.content) do
    if child.t == "Div" and child.classes:includes("value-box") then
      for _, name in ipairs(INHERITABLE_ATTRS) do
        -- nil (not falsy) is "unset": an explicit blank value like
        -- icon-color="" is a deliberate override and must not be replaced by
        -- the row's default, same principle as the icon_size_raw ~= "" guard
        -- further down.
        if child.attributes[name] == nil and el.attributes[name] ~= nil then
          child.attributes[name] = el.attributes[name]
        end
      end
    end
  end
  return el
end

function Div(el)
  if el.classes:includes("value-box") then

    -- Existing attributes.
    --
    -- Every value below that ends up interpolated into an HTML attribute is
    -- escaped at the point it's read, EXCEPT: icon (kept raw — it's used for
    -- file I/O via io.open() and pattern-matched against literal prefixes
    -- like "fa-"/"ph-" by detect_icon_type(); a separate icon_attr below
    -- holds the escaped form for the class=/src= sites that actually need
    -- it), value/title (interpolated into element *content*, not an
    -- attribute — raw HTML there is a documented feature, not an oversight;
    -- see the README note on markdown not being processed in either), and
    -- delta (also content, but escaped rather than raw HTML — kept raw at
    -- read time only because it's pattern-matched for sign inference below;
    -- see escape_attr(delta) at the point it's interpolated, further down).
    -- Escaping at read time is safe for every attribute compared against a
    -- known keyword (align, valign, icon-position, ...): escape_attr() is a
    -- no-op unless the value contains & < > ", and a value containing any of
    -- those was never going to match a plain keyword like "left" anyway.
    local icon      = el.attributes["icon"] or ""
    local icon_attr = escape_attr(icon)
    local icon_type -- supports "fa" | "bi" | "svg" | "png" | "material" | "material-outlined" | "material-rounded" | "material-sharp" | "tabler" | "phosphor"
    if el.attributes["icon-type"] then
      icon_type = el.attributes["icon-type"]
    elseif icon ~= "" then
      icon_type = detect_icon_type(icon)
    else
      icon_type = "bi"
    end
    local color_raw = el.attributes["color"] or "bg-blue"
    -- A leading #, rgb(/rgba(, hsl(/hsla(, or var( means this is a literal CSS
    -- colour value, not one of the prespecified bg-* classes, and must become
    -- an inline background-color instead of a class name (see README "CSS
    -- class or value" note on color) — a raw value dropped into class="..."
    -- matches no stylesheet rule and silently does nothing.
    local color_is_value = color_raw:lower():match("^#") or color_raw:lower():match("^rgba?%(")
      or color_raw:lower():match("^hsla?%(") or color_raw:lower():match("^var%(")
    local color       = escape_attr(color_raw)
    local color_class = color_is_value and "" or color
    local value     = el.attributes["value"] or ""

    -- Geometry: each emitted as a --vb-* custom property only when set, so an
    -- unset box takes value-box.css's own var(--vb-*, <default>) fallback —
    -- which a project can retarget on :root. Keep these defaults and the CSS
    -- ones in sync.
    local width_style      = themed_decl(el.attributes["width"], "--vb-width")
    local height_style     = themed_decl(el.attributes["height"], "--vb-height")
    local min_height_style = themed_decl(el.attributes["min-height"], "--vb-min-height")
    local padding_style    = themed_decl(el.attributes["padding"], "--vb-padding")

    -- align resolves to "left" for the icon optical-bearing logic and the
    -- icon_align_value mapping below, but the two custom properties it drives
    -- are only emitted when align was actually set (so an unset box inherits
    -- the stylesheet default, themeable via --vb-text-align / --vb-icon-align).
    local align_raw = el.attributes["align"]
    local align     = escape_attr(align_raw or "left")
    -- The icon (and, in the row layout, .vb-content) are flex items, not
    -- inline content of .value-box, so text-align alone can't position
    -- them — see --vb-icon-align below. Same left/center/right shorthand
    -- convention as valign_map further down, with a raw-value fallback for
    -- anyone who wants to pass a CSS keyword directly.
    local align_map = { left = "flex-start", center = "center", right = "flex-end" }
    local icon_align_value = align_map[align] or align
    local align_style = ""
    if align_raw ~= nil and align_raw ~= "" then
      align_style = css_decl("--vb-text-align", align) .. css_decl("--vb-icon-align", icon_align_value)
    end
    local valign_raw = el.attributes["valign"]
    local valign     = escape_attr(valign_raw or "middle")
    local href      = escape_attr(el.attributes["href"] or "")
    -- Only meaningful alongside href — a target on a plain div has nothing to
    -- navigate. target="_blank" also gets rel="noopener noreferrer" added
    -- automatically: an opener-less new tab is the whole point of _blank, and
    -- leaving window.opener reachable is a known phishing vector (reverse
    -- tabnabbing) that every other target value doesn't share.
    --
    -- Also sets data-preview-link="false" whenever a target is given.
    -- Reveal.js's previewLinks option (common in slide decks) hijacks the
    -- click on *every* http(s) anchor and opens the href in an in-slide
    -- iframe overlay instead — regardless of the anchor's own target
    -- attribute, so target="_blank" alone silently does nothing under
    -- previewLinks:true, and the overlay comes up blank for any site that
    -- blocks framing (e.g. YouTube). data-preview-link="false" is the one
    -- attribute reveal.js itself checks to exclude an anchor from that
    -- hijack, so setting target is treated as an explicit request for real
    -- link semantics.
    local target    = escape_attr(el.attributes["target"] or "")
    local target_attr = ""
    if href ~= "" and target ~= "" then
      target_attr = string.format(' target="%s" data-preview-link="false"', target)
      if target == "_blank" then
        target_attr = target_attr .. ' rel="noopener noreferrer"'
      end
    end
    local icon_pos  = el.attributes["icon-position"] or "top" -- "top" | "bottom" | "left" | "right"
    local value_pos = el.attributes["value-position"] or "top" -- "top" | "bottom" | "left" | "right"
    -- Font sizes and text colours. Each is written to its element's inline
    -- style only when set — otherwise value-box.css's own
    -- `<prop>: var(--vb-*, <default>)` rule supplies the default, so a project
    -- can retarget it on :root. value/title colour fall back to font-color;
    -- that chain now lives in the stylesheet
    -- (color: var(--vb-value-color, var(--vb-font-color, white))), so passing
    -- font-color raw here — not the resolved white — is deliberate.
    local font_color_raw = el.attributes["font-color"]
    local font_size_style       = themed_decl(el.attributes["font-size"], "font-size")
    local value_font_size_style = themed_decl(el.attributes["value-font-size"], "font-size")
    local font_color_style  = themed_decl(font_color_raw, "color")
    local value_color_style = themed_decl(el.attributes["value-color"] or font_color_raw, "color")
    local icon_color_style  = themed_decl(el.attributes["icon-color"], "color")

    -- A short label rendered above the value. title-font-size / title-color are
    -- deliberately left unset by default so the stylesheet's .value-box .title
    -- rule (and its var() fallbacks) supply them; setting the attribute wins.
    local title = el.attributes["title"] or ""
    local title_color_style     = themed_decl(el.attributes["title-color"] or font_color_raw, "color")
    local title_font_size_style = themed_decl(el.attributes["title-font-size"], "font-size")

    -- Delta / trend indicator: a small badge next to the value, e.g. "+12%"
    -- with an arrow. delta-direction picks the arrow glyph; when unset it's
    -- inferred from a leading "+"/"-" in the delta text, which covers the
    -- common case without direction becoming another required attribute.
    -- Colour is left to inherit by default rather than mapped from
    -- direction — "up" isn't always good news (a falling cost, say), so
    -- this filter doesn't guess; set delta-color explicitly for that.
    -- delta itself is escaped rather than treated as raw HTML content like
    -- value/title — see the CONTRIBUTING.md note on not growing that list.
    local delta = el.attributes["delta"] or ""
    -- Matched case-insensitively for the same reason passthrough attribute
    -- names are: HTML authors reach for "Up"/"Down" as often as not, and
    -- there is no reason to make that fail silently.
    local delta_direction = el.attributes["delta-direction"] -- "up" | "down" | "flat" | nil
    if delta_direction then
      delta_direction = delta_direction:lower()
    end
    if not delta_direction and delta ~= "" then
      if delta:match("^%+") then
        delta_direction = "up"
      elseif delta:match("^%-") then
        delta_direction = "down"
      end
    end
    -- "flat" uses an arrow rather than a dash so all three glyphs read as
    -- one family (a rising/falling/level line) rather than mixing shapes.
    local delta_arrows = { up = "▲", down = "▼", flat = "→" }
    local delta_arrow = delta_arrows[delta_direction or ""] or ""
    if delta_direction and delta_direction ~= "" and not delta_arrows[delta_direction] then
      io.stderr:write(string.format("value-box warning: unrecognised delta-direction '%s' — expected up, down or flat; showing no arrow\n", delta_direction))
    end
    local delta_color_style = themed_decl(el.attributes["delta-color"], "color")
    local delta_font_size_style = themed_decl(el.attributes["delta-font-size"], "font-size")
    local delta_extra_style = escape_attr(el.attributes["delta-extra-style"] or "")
    -- Escape hatch for the value+delta row wrapper itself (see below), kept
    -- consistent with every other structural element this filter builds.
    local value_row_extra_style = escape_attr(el.attributes["value-row-extra-style"] or "")

    local icon_size_raw = el.attributes["icon-size"]

    -- em units for font-based icons (BI/FA), px for image-based (SVG/PNG).
    -- An empty string is truthy in Lua, so icon-size="" must be treated as
    -- unset explicitly or it would suppress the default entirely. Escaped
    -- after resolving rather than at el.attributes read time, since
    -- icon_size_raw can be Lua nil (no "or" fallback) and escape_attr(nil)
    -- would turn that into the literal string "nil" via tostring().
    --
    -- Font-glyph icons: emit nothing when unset — value-box.css's
    -- .value-box .icon { font-size: var(--vb-icon-size, 3em) } supplies the
    -- default and lets a project retarget it. Image icons still need an
    -- explicit pixel size on the element (they carry no font-size), so their
    -- 128px default stays here.
    local icon_size_font  -- nil unless icon-size was set
    local icon_size_img
    if icon_size_raw and icon_size_raw ~= "" then
      icon_size_font = escape_attr(icon_size_raw)
      icon_size_img  = escape_attr(icon_size_raw)
    else
      icon_size_font = nil
      icon_size_img  = "128px"
    end

    -- Fragment logic
    local fragment_attr = el.attributes["fragment"]
    local fragment_class = ""
    if fragment_attr then
      local effect = (fragment_attr == "true") and "fade-in-then-semi-out" or fragment_attr
      fragment_class = " fragment " .. escape_attr(effect)
    end

    -- Fragment Index logic
    local index_attr = el.attributes["index"]
    local index_data = ""
    if index_attr then
      index_data = string.format(' data-fragment-index="%s"', escape_attr(index_attr))
    end

    -- Escape hatches: raw CSS text a user can inject into each generated
    -- element's own style attribute, verbatim, alongside whatever this
    -- filter itself declares there.
    local outer_extra_style   = escape_attr(el.attributes["outer-extra-style"] or "")
    local icon_extra_style    = escape_attr(el.attributes["icon-extra-style"] or "")
    local content_extra_style = escape_attr(el.attributes["content-extra-style"] or "")
    local details_extra_style = escape_attr(el.attributes["details-extra-style"] or "")
    local value_extra_style   = escape_attr(el.attributes["value-extra-style"] or "")
    local title_extra_style   = escape_attr(el.attributes["title-extra-style"] or "")

    -- Preserve the id, plus any classes and attributes this filter does not
    -- itself interpret, so a value-box behaves like an ordinary div for
    -- anything outside its own vocabulary: revealjs auto-animate matching via
    -- data-id, crossref targets via #id, custom per-box classes, and ARIA
    -- attributes all keep working even though the div is replaced with raw
    -- HTML below rather than going through Pandoc's own div-to-HTML writer.
    -- A literal data-fragment-index is only a real collision when index is
    -- actually set — that's what generates this filter's own
    -- data-fragment-index. Without an index, a literal data-fragment-index is
    -- just an ordinary data-* attribute and passes through untouched.
    local id_attr, extra_classes, passthrough_attrs = build_wrapper_attrs(el, "value-box", "outer-extra-style", {
      name = "data-fragment-index",
      active = index_attr ~= nil,
      warning = "value-box warning: a literal 'data-fragment-index' attribute is ignored to avoid clashing with the one generated from the index attribute\n",
    })

    -- icon-position controls the outer wrapper: icon vs. everything else.
    -- The actual flex rules (row direction, gap, icon/content sizing) live in
    -- value-box.css keyed off these classes — nothing here is a CSS value,
    -- just a switch.
    local outer_layout_class = ""
    local icon_row = (icon_pos == "left" or icon_pos == "right")
    if icon_pos == "left" then
      outer_layout_class = " vb-icon-left"
    elseif icon_pos == "right" then
      outer_layout_class = " vb-icon-right"
    end

    -- value-position controls the inner content wrapper: value vs. details,
    -- independent of icon-position. Kept separate from the icon-position
    -- class because a title has to sit above this row rather than become a
    -- third item in it — see the content wrapper below, which moves this
    -- class onto an inner element when both apply.
    local value_pos_class = ""
    if value_pos == "left" then
      value_pos_class = " vb-value-left"
    elseif value_pos == "right" then
      value_pos_class = " vb-value-right"
    end

    -- Compensate for icon-font glyphs' built-in optical bearing so a stacked
    -- icon visually lines up with the left/right edge of the value/details
    -- text. Only meaningful for font-glyph icons (fa/bi/tabler/phosphor/
    -- material) stacked top/bottom — SVG/PNG icons have no such bearing, and
    -- left/right icon-position already aligns on the cross axis instead.
    local icon_bearing_class = ""
    if (icon_pos == "top" or icon_pos == "bottom") and (icon_type == "fa" or icon_type == "bi" or icon_type == "tabler" or icon_type == "phosphor" or material_variants[icon_type]) then
      if align == "left" then
        icon_bearing_class = " vb-bearing-left"
      elseif align == "right" then
        icon_bearing_class = " vb-bearing-right"
      end
    end

    -- Vertical alignment. When the icon has already put the wrapper into a
    -- row (icon-position left/right), valign controls the cross-axis
    -- align-items; otherwise it controls the column's main-axis
    -- justify-content. Either way this is just the one custom property the
    -- corresponding CSS rule reads — see value-box.css. Emitted only when
    -- valign was set, so an unset box takes the stylesheet's own centred
    -- default (var(--vb-justify-content, center) / var(--vb-align-items,
    -- center)), themeable per project on :root.
    local outer_layout_style = ""
    if valign_raw ~= nil and valign_raw ~= "" then
      local valign_map = { top = "flex-start", middle = "center", bottom = "flex-end" }
      local align_value = valign_map[valign] or valign  -- fall back to raw value if not a shorthand
      if icon_row then
        outer_layout_style = css_decl("--vb-align-items", align_value)
      else
        outer_layout_style = css_decl("--vb-justify-content", align_value)
      end
    end

    -- Assemble the outer wrapper's custom properties. Each piece was built
    -- above with themed_decl / an explicit "was it set?" check, so it's the
    -- empty string for any attribute this box (and its row) left unset — and
    -- value-box.css's var(--vb-*, <default>) fallback covers that case, which
    -- is also what makes every default themeable on :root. A raw colour value
    -- (color="#...") is the one that still always lands inline, since there's
    -- no sensible stylesheet default for it.
    local base_style =
      width_style ..
      height_style ..
      min_height_style ..
      padding_style ..
      align_style ..
      (color_is_value and css_decl("--vb-bg", color) or "") ..
      outer_layout_style

    -- Classes: value-box's own name, then this box's layout switches, then
    -- colour/fragment/user classes — kept in this order so the generated
    -- class list reads structural-first, same as the style properties above.
    local outer_class = "value-box" .. outer_layout_class .. " " .. color_class .. fragment_class .. extra_classes

    local html_open
    if href ~= "" then
      html_open = string.format(
        '<a%s href="%s"%s class="%s" style="%s%s"%s%s>',
        id_attr, href, target_attr, outer_class, base_style, outer_extra_style, index_data, passthrough_attrs
      )
    else
      html_open = string.format(
        '<div%s class="%s" style="%s%s"%s%s>',
        id_attr, outer_class, base_style, outer_extra_style, index_data, passthrough_attrs
      )
    end

    -- Build icon HTML (empty string if no icon).
    -- Every icon-font branch shares the same style prelude. font-size and
    -- color are emitted only when icon-size / icon-color were set; otherwise
    -- the .value-box .icon rule in value-box.css (font-size: var(--vb-icon-
    -- size, 3em); color: var(--vb-icon-color, white)) supplies them.
    local icon_font_style = css_decl("font-size", icon_size_font)
      .. icon_color_style .. icon_extra_style

    local icon_html = ""
    if icon ~= "" then
      if icon_type == "fa" then
        include_icon_stylesheet(FONT_AWESOME_CSS)
        icon_html = string.format('<i class="icon%s %s" style="%s"></i>', icon_bearing_class, icon_attr, icon_font_style)

      elseif icon_type == "svg" then
        local svg_file = io.open(icon, "r")
        if svg_file then
          local svg_content = svg_file:read("*all")
          svg_file:close()
          local natural_w, natural_h = svg_dimensions(svg_content)
          local svg_size_style = icon_size_style(icon_size_img, natural_w, natural_h)
          svg_content = svg_content:gsub('<svg', string.format('<svg style="%s"', svg_size_style))
          icon_html = string.format(
            '<span class="icon" style="display:inline-flex; align-items:center; justify-content:center; font-size:inherit;%s">%s</span>',
            icon_extra_style, svg_content
          )
        else
          io.stderr:write(string.format("value-box warning: SVG file not found: %s\n", icon))
        end

      elseif icon_type == "png" then
        local png_file = io.open(icon, "r")
        if png_file then
          png_file:close()
          local natural_w, natural_h = png_dimensions(icon)
          local png_size_style, is_fallback_square = icon_size_style(icon_size_img, natural_w, natural_h)
          local object_fit_style = is_fallback_square and "object-fit:contain; " or ""
          icon_html = string.format(
            '<img class="icon" src="%s" style="%s%sdisplay:block;%s" alt="">',
            icon_attr, png_size_style, object_fit_style, icon_extra_style
          )
        else
          io.stderr:write(string.format("value-box warning: PNG file not found '%s', falling back to Bootstrap Icons\n", icon))
          include_icon_stylesheet(BOOTSTRAP_ICONS_CSS)
          icon_html = string.format('<i class="icon%s bi %s" style="%s"></i>', icon_bearing_class, icon_attr, icon_font_style)
        end

      elseif material_variants[icon_type] then
        local variant = material_variants[icon_type]
        include_icon_stylesheet(string.format(MATERIAL_SYMBOLS_CSS_FMT, variant.family))
        icon_html = string.format(
          '<span class="icon%s %s" style="%s">%s</span>',
          icon_bearing_class, variant.class, icon_font_style, icon_attr
        )

      elseif icon_type == "tabler" then
        include_icon_stylesheet(TABLER_ICONS_CSS)
        icon_html = string.format('<i class="icon%s ti %s" style="%s"></i>', icon_bearing_class, icon_attr, icon_font_style)

      elseif icon_type == "phosphor" then
        local weight_token = icon:match("^(%S+)")
        local weight_dir = phosphor_weight_dirs[weight_token] or "regular"
        include_icon_stylesheet(string.format(PHOSPHOR_ICONS_CSS_FMT, weight_dir))
        icon_html = string.format('<i class="icon%s %s" style="%s"></i>', icon_bearing_class, icon_attr, icon_font_style)

      else
        -- Bootstrap Icons (default)
        include_icon_stylesheet(BOOTSTRAP_ICONS_CSS)
        icon_html = string.format('<i class="icon%s bi %s" style="%s"></i>', icon_bearing_class, icon_attr, icon_font_style)
      end
    end

    -- Build value HTML (empty string if no value)
    local value_html = ""
    if value ~= "" then
      value_html = string.format(
        '<div class="value" style="%s%s%s">%s</div>',
        value_font_size_style, value_color_style,
        value_extra_style, value
      )
    end

    -- Build delta HTML (empty string if no delta) and, if present, wrap it
    -- alongside the value in a row so the two sit side by side wherever
    -- value_html ends up placed (top/bottom/left/right value-position all
    -- inject value_html the same way, so wrapping it here covers every case).
    -- The row's own flex rules are static and live in value-box.css under
    -- .vb-value-row; when value-position is left/right, that same rule also
    -- protects the row from shrinking (see .vb-value-left/.vb-value-right
    -- .vb-value-row there).
    if delta ~= "" then
      local arrow_html = ""
      if delta_arrow ~= "" then
        arrow_html = string.format('<span class="delta-arrow" aria-hidden="true">%s</span> ', delta_arrow)
      end
      local delta_html = string.format(
        '<div class="delta" style="%s%s%s">%s%s</div>',
        delta_font_size_style, delta_color_style,
        delta_extra_style, arrow_html, escape_attr(delta)
      )
      value_html = string.format(
        '<div class="vb-value-row" style="%s">%s%s</div>',
        value_row_extra_style, value_html, delta_html
      )
    end

    -- For bottom placement, defer icon injection; otherwise inject it now
    if icon_pos ~= "bottom" then
      html_open = html_open .. icon_html
    end

    -- Build title HTML (empty string if no title)
    local title_html = ""
    if title ~= "" then
      title_html = string.format(
        '<div class="title" style="%s%s%s">%s</div>',
        title_font_size_style, title_color_style,
        title_extra_style, title
      )
    end

    -- A left/right value-position turns the value and details into a flex row.
    -- A title has to sit above that row rather than become a third item in it,
    -- so when both are in play the row class moves to an inner wrapper and
    -- .vb-content becomes the column that stacks title above row. With no
    -- title, or no row, the markup is unchanged.
    local use_row_wrapper = (title ~= "" and value_pos_class ~= "")

    -- Open the content wrapper (holds title, value and details, positioned
    -- independently of the icon)
    html_open = html_open .. string.format('<div class="vb-content%s" style="%s">',
      use_row_wrapper and "" or value_pos_class, content_extra_style)

    html_open = html_open .. title_html

    if use_row_wrapper then
      html_open = html_open .. string.format('<div class="vb-row%s">', value_pos_class)
    end

    -- For bottom placement, defer value injection; otherwise inject it now
    if value_pos ~= "bottom" then
      html_open = html_open .. value_html
    end

    -- Open the details wrapper
    html_open = html_open .. string.format('<div class="details" style="%s%s%s">',
      font_size_style, font_color_style, details_extra_style)

    -- Close details, optionally append value below, close the row wrapper if
    -- one was opened, close content wrapper, optionally append icon below,
    -- then close outer
    local html_close = '</div>' -- close .details
    if value_pos == "bottom" then
      html_close = html_close .. value_html
    end
    if use_row_wrapper then
      html_close = html_close .. '</div>' -- close .vb-row
    end
    html_close = html_close .. '</div>' -- close .vb-content
    if icon_pos == "bottom" then
      html_close = html_close .. icon_html
    end
    html_close = html_close .. (href ~= "" and '</a>' or '</div>')

    local result = pandoc.List({pandoc.RawBlock("html", html_open)})
    result:extend(el.content)
    result:insert(pandoc.RawBlock("html", html_close))

    include_value_box_stylesheet()

    return result

  elseif el.classes:includes("value-box-row") then

    -- Lays child value boxes out with equal width and equal height — the
    -- single most common value-box layout, and previously only achievable by
    -- hand-rolling Quarto's own .columns/.column scaffolding and hand-setting
    -- height on every box.
    --
    -- No columns attribute set: a single flex row, one column per child, no
    -- wrapping — the default in value-box.css, so nothing further to do here.
    -- columns="N" set: switches to CSS Grid via the vb-row-grid class, so
    -- extra children beyond N wrap onto further rows automatically (Grid's
    -- default auto-flow), with grid-auto-rows:1fr (also in the stylesheet)
    -- keeping wrapped rows equal height too. The column count itself is the
    -- one part that varies per box, so it travels as a custom property
    -- rather than being baked into the class.
    local columns_attr = el.attributes["columns"]
    local row_layout_class = ""
    local columns_style = ""
    if columns_attr and columns_attr ~= "" then
      local columns_n = tonumber(columns_attr)
      if columns_n then
        row_layout_class = " vb-row-grid"
        columns_style = css_decl("--vb-row-columns", tostring(columns_n))
      else
        io.stderr:write(string.format(
          "value-box warning: unrecognised columns value '%s' — expected a number; falling back to a single row\n",
          columns_attr))
      end
    end

    -- responsive="false" opts out of the wrap / column-drop behaviour and
    -- restores the pre-1.6 rigid layout via the vb-row-fixed class. Matched
    -- case-insensitively; "" and "true" keep the default, anything else warns
    -- and keeps the default — same validate-or-warn shape as the columns block.
    local responsive_attr = el.attributes["responsive"]
    if responsive_attr then
      local responsive_lower = responsive_attr:lower()
      if responsive_lower == "false" then
        row_layout_class = row_layout_class .. " vb-row-fixed"
      elseif responsive_lower ~= "" and responsive_lower ~= "true" then
        io.stderr:write(string.format(
          "value-box warning: unrecognised responsive value '%s' — expected true or false; keeping the responsive default\n",
          responsive_attr))
      end
    end

    -- min-column-width and gap: how narrow a column may get before the row
    -- wraps (flex) / drops a column (grid), and the space between boxes. Each
    -- emitted only when set — value-box.css carries the defaults as var()
    -- fallbacks (--vb-row-min-col 14rem, --vb-row-gap 1.5rem), so a plain row
    -- carries neither, and a project can retarget both on :root.
    local min_col_style = themed_decl(el.attributes["min-column-width"], "--vb-row-min-col")
    local gap_style     = themed_decl(el.attributes["gap"], "--vb-row-gap")
    local row_extra_style = escape_attr(el.attributes["extra-style"] or "")

    local id_attr, extra_classes, passthrough_attrs = build_wrapper_attrs(el, "value-box-row", "extra-style", nil)

    local html_open = string.format(
      '<div%s class="value-box-row%s%s" style="%s%s%s%s"%s>',
      id_attr, row_layout_class, extra_classes,
      columns_style, gap_style, min_col_style, row_extra_style, passthrough_attrs
    )

    local result = pandoc.List({pandoc.RawBlock("html", html_open)})
    result:extend(el.content)
    result:insert(pandoc.RawBlock("html", "</div>"))

    include_value_box_stylesheet()

    return result
  end

end

-- Two sequential passes: inject_row_defaults merges row-level attribute
-- defaults into each .value-box child while it's still a real Div, then Div
-- (unchanged) runs its own full pass over the resulting document to expand
-- both .value-box and .value-box-row into HTML.
return {
  { Div = inject_row_defaults },
  { Div = Div },
}
