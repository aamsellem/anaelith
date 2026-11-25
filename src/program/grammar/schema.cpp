// src/program/grammar/schema.cpp - see schema.hpp. The JSON-schema conversion is a port of llama.cpp's
// json-schema-to-grammar (schema_convert.cpp, json_schema.cpp); this file adds the tool-call and request grammars.
#include "basalt/program/grammar/schema.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <limits>
#include <tuple>
#include <map>
#include <set>
#include <stdexcept>

#include "schema_internal.hpp"

namespace basalt::program::grammar {

namespace {

using Type = JsonValue::Type;

std::string literal(const std::string& s) { return gbnf_format_literal(s); }

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) o += (i ? sep : "") + v[i];
    return o;
}

bool read_json(const std::string& text, JsonValue& v, std::string& err) { return JsonReader::parse(text, v, err); }

// Follows local $ref links ("#/a/b") to the schema they name; null if one does not resolve.
const JsonValue* deref(const JsonValue* root, const JsonValue* s) {
    for (int guard = 0; s && s->is(Type::Object) && guard < 32; ++guard) {
        const JsonValue* r = s->get("$ref");
        if (!r || !r->is(Type::String)) break;
        if (r->s.compare(0, 2, "#/") != 0 || !root) return nullptr;
        const JsonValue* cur = root;
        size_t pos = 2;
        while (cur && pos <= r->s.size()) {
            size_t e = r->s.find('/', pos);
            std::string part = r->s.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
            if (cur->is(Type::Object)) {
                cur = cur->get(part);
            } else if (cur->is(Type::Array)) {
                const size_t limit = cur->items.size();
                bool ok_idx = !part.empty() && limit > 0;
                size_t idx = 0;
                for (size_t k = 0; ok_idx && k < part.size(); ++k) {
                    const char ch = part[k];
                    if (ch < '0' || ch > '9') { ok_idx = false; break; }
                    const size_t d = (size_t) (ch - '0');
                    if (idx > ((std::numeric_limits<size_t>::max)() - d) / 10) { ok_idx = false; break; }
                    idx = idx * 10 + d;
                    if (idx >= limit) { ok_idx = false; break; }
                }
                cur = ok_idx ? &cur->items[idx] : nullptr;
            } else {
                cur = nullptr;
            }
            if (e == std::string::npos) break;
            pos = e + 1;
        }
        s = cur;
    }
    if (s && s->is(Type::Object)) {
        if (const JsonValue* r = s->get("$ref"); r && r->is(Type::String)) return nullptr;
    }
    return s;
}

}  // namespace

std::string json_schema_to_gbnf(const std::string& schema_json, std::string& err) {
    JsonValue schema;
    if (!read_json(schema_json, schema, err)) return "";
    try {
        return schema_document_to_gbnf(build_schema_document(schema));
    } catch (const std::invalid_argument& e) {
        err = std::string("schema: ") + e.what();
    } catch (const std::runtime_error& e) {
        err = std::string("schema: JSON schema conversion failed:\n") + e.what();
    }
    return "";
}

// ------------------------------------------------------------------------------------------ tool calls

namespace {

// True when the schema is a type that is raw text in a parameter: "string", or a type list naming "string".
bool is_text_type(const JsonValue& t) {
    if (t.is(Type::String)) return t.s == "string";
    return t.is(Type::Array) && std::any_of(t.items.begin(), t.items.end(),
                                            [](const JsonValue& v) { return v.is(Type::String) && v.s == "string"; });
}

// One branch of an anyOf/oneOf/allOf: a plain string (type "string" or a list of string and null, no enum/const/
// structure) or null. `has_text` is set when it can be text.
bool plain_text_branch(const JsonValue* b, bool& has_text) {
    if (!b || !b->is(Type::Object)) return false;
    for (const char* k : {"enum", "const", "anyOf", "oneOf", "allOf", "properties", "items", "$ref"})
        if (b->get(k)) return false;
    const JsonValue* t = b->get("type");
    if (!t) return false;
    if (t->is(Type::String)) {
        if (t->s == "string") has_text = true;
        return t->s == "string" || t->s == "null";
    }
    if (!t->is(Type::Array)) return false;
    for (const JsonValue& v : t->items) {
        if (!v.is(Type::String) || (v.s != "string" && v.s != "null")) return false;
        if (v.s == "string") has_text = true;
    }
    return true;
}

// A parameter written as anyOf/oneOf (or a one-branch allOf) whose every branch is a plain string or null, with at
// least one string: it takes raw text like a string, and `null` is just text it may write.
bool text_union(const JsonValue* ps, const JsonValue* root) {
    if (!ps || !ps->is(Type::Object) || ps->get("type") || ps->get("enum") || ps->get("const") ||
        ps->get("properties") || ps->get("items"))
        return false;
    const JsonValue* u = nullptr;
    for (const char* k : {"anyOf", "oneOf", "allOf"}) {
        if (const JsonValue* v = ps->get(k)) {
            if (u || !v->is(Type::Array) || v->items.empty()) return false;
            if (k[0] == 'a' && k[1] == 'l' && v->items.size() != 1) return false;
            u = v;
        }
    }
    if (!u) return false;
    bool has_text = false;
    for (const JsonValue& b : u->items)
        if (!plain_text_branch(deref(root, &b), has_text)) return false;
    return has_text;
}

std::string hex_escape(unsigned char c) {
    static const char* h = "0123456789abcdef";
    return std::string("\\x") + h[c >> 4] + h[c & 15];
}

// The UTF-8 characters of `s`: a grammar's literals and classes read characters, not bytes.
std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        while (i + n < s.size() && ((unsigned char) s[i + n] & 0xC0) == 0x80) ++n;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

namespace {
bool is_space(const std::string& ch) { return ch == " " || ch == "\t" || ch == "\r" || ch == "\n"; }

std::string trimmed(const std::string& t) {
    size_t b = 0, e = t.size();
    while (b < e && std::strchr(" \t\r\n", t[b]) != nullptr) ++b;
    while (e > b && std::strchr(" \t\r\n", t[e - 1]) != nullptr) --e;
    return t.substr(b, e - b);
}

// One string a parameter's free text may not contain, as characters; `gap`, when not negative, is the element
// before which any run of white space may stand (where the closing tag ends), `gap2` a second such place.  `quoted`:
// the string is excluded after a quoted parameter's tag as well.
struct TextEnd {
    std::vector<std::string> chars;
    int gap = -1;
    int gap2 = -1;
    bool quoted = false;
};

// The strings a parameter's free text may not contain.
//
// A form whose parser ends a value at the first closing tag (text_holds_end off) excludes the tag itself.
// Otherwise a value may hold the tag: an agent that edits tool-call markup writes one inside its argument, and the
// server's parser (serve/frontend.py param_end) ends a value only where the call's structure follows a tag.  So the
// text may not hold the tag followed, after any white space, by what a call writes after a parameter: another
// parameter or the function's end.  The call's end and a new call are excluded there as well, though the parser would
// read them as text: a call that skips the function's end is then refused instead of swallowed by the value.
// After the tag of a parameter the value quotes, the text may go on with anything (add_free_text) but the function's
// end with the call's end or a new call behind it: that is a call ending, not a quoted line.
std::vector<TextEnd> free_text_ends(const ToolForm& form) {
    std::vector<TextEnd> ends;
    const std::vector<std::string> tag = utf8_chars(form.text_end);
    if (form.text_holds_end && !tag.empty() && form.param_close.find(form.text_end) != std::string::npos) {
        std::set<std::string> seen;
        for (const std::string& next : {form.param_open, form.fn_close, trimmed(form.call_close), trimmed(form.call_open)}) {
            if (next.empty() || !seen.insert(next).second) continue;
            TextEnd e;
            e.chars = tag;
            e.gap = (int) tag.size();
            for (const std::string& ch : utf8_chars(next)) e.chars.push_back(ch);
            ends.push_back(e);
        }
        for (const std::string& next : {trimmed(form.call_close), trimmed(form.call_open)}) {
            if (form.fn_close.empty() || next.empty()) continue;
            TextEnd e;
            e.chars = tag;
            e.gap = (int) tag.size();
            for (const std::string& ch : utf8_chars(form.fn_close)) e.chars.push_back(ch);
            e.gap2 = (int) e.chars.size();
            for (const std::string& ch : utf8_chars(next)) e.chars.push_back(ch);
            e.quoted = true;
            ends.push_back(e);
        }
    }
    if (ends.empty()) ends.push_back(TextEnd{tag, -1});
    return ends;
}

// A call quoted inside a value, as the parser counts one (`<function=NAME>` then a parameter or the function's end):
// up to its own function end the structure may follow a closing tag, since it is the quoted call's.
std::string add_quoted_call(SchemaConverter& c, const ToolForm& form) {
    const std::string mid = trimmed(form.fn_mid);
    if (!form.text_holds_end || form.fn_open.empty() || form.fn_close.empty() || form.param_open.empty() || mid.empty())
        return "";
    const std::vector<std::string> chars = utf8_chars(form.fn_close);
    const size_t n = chars.size();
    const std::set<std::string> alpha(chars.begin(), chars.end());
    std::string excluded;
    for (const std::string& a : alpha) excluded += a.size() == 1 ? hex_escape((unsigned char) a[0]) : a;
    for (size_t i = 0; i < n; ++i) {   // the text up to and with the function's end: its KMP automaton, ended by the match
        std::vector<std::string> alts;
        for (const std::string& a : alpha) {
            std::vector<std::string> seen(chars.begin(), chars.begin() + (std::ptrdiff_t) i);
            seen.push_back(a);
            size_t k = (std::min)(seen.size(), n);
            while (k > 0 && !std::equal(chars.begin(), chars.begin() + (std::ptrdiff_t) k, seen.end() - (std::ptrdiff_t) k)) --k;
            alts.push_back(k == n ? literal(a) : literal(a) + " quoted-text-" + std::to_string(k));
        }
        alts.push_back("[^" + excluded + "] quoted-text-0");
        c.add_rule("quoted-text-" + std::to_string(i), join(alts, " | "));
    }
    c.add_rule("quoted-name", "[^\\x3c\\x3e\\x20\\x09\\x0d\\x0a]+");
    c.add_rule("quoted-space", "[\\x20\\x09\\x0d\\x0a]*");
    return c.add_rule("quoted-call", literal(form.fn_open) + " quoted-name " + literal(mid) + " quoted-space (" +
                                         literal(form.param_open) + " quoted-text-0 | " + literal(form.fn_close) + ")");
}
}  // namespace

// Text that never contains one of the form's excluded strings (free_text_ends), with quoted calls passed whole: the
// automaton that reads every string at once, one rule per state.  A state is the set of places reached in the
// strings; a move that would complete one of them is the move the automaton refuses, and the text may stop in any
// state.  With the closing tag as the one string this is its KMP automaton, state i the matched length.
//
// Where the text may hold the tag, the automaton also follows the parameters the value quotes
// (`<parameter=NAME>`): the tag that closes a quoted one is the quoted parameter's, and what stands after it is the
// value's whatever it is - the lines of a call that follow a quoted parameter in a file, another quoted parameter.
// The parser reads it the same way (serve/frontend.py param_close): the structure after such a tag ends the value
// only where the call can take it.  Up to kQuoteDepth quoted parameters are counted.
std::string add_free_text(SchemaConverter& c, const ToolForm& form) {
    constexpr int kQuoteDepth = 2;
    const std::vector<TextEnd> ends = free_text_ends(form);
    struct State {
        std::set<std::pair<int, int>> at;   // (string, characters matched)
        int open = 0;                       // characters of a quoted opening read; past its length: in the name
        int depth = 0;                      // quoted parameters open
        bool operator<(const State& o) const { return std::tie(at, open, depth) < std::tie(o.at, o.open, o.depth); }
    };
    std::set<std::string> alpha;
    bool gaps = false;
    for (const TextEnd& e : ends) {
        alpha.insert(e.chars.begin(), e.chars.end());
        gaps = gaps || e.gap >= 0;
    }
    if (gaps)
        for (const char* ws : {" ", "\t", "\r", "\n"}) alpha.insert(ws);
    const std::vector<std::string> popen = gaps ? utf8_chars(form.param_open) : std::vector<std::string>();
    const int P = (int) popen.size();
    if (P > 0) {
        alpha.insert(popen.begin(), popen.end());
        alpha.insert("<");
        alpha.insert(">");
    }
    const auto start = [&] {
        std::set<std::pair<int, int>> s;
        for (int i = 0; i < (int) ends.size(); ++i) s.insert({i, 0});
        return s;
    };
    // `ch` empty: any character outside the alphabet
    const auto move = [&](const State& from, const std::string& ch) {
        State to;
        to.at = start();
        to.depth = from.depth;
        if (P > 0) {
            const int again = !ch.empty() && ch == popen[0] ? 1 : 0;
            if (from.open >= P) {   // the name: anything but a bracket or white space
                if (ch.empty() || !(ch == "<" || ch == ">" || is_space(ch))) to.open = P + 1;
                else if (ch == ">" && from.open > P) to.depth = (std::min)(from.depth + 1, kQuoteDepth);
                else to.open = again;
            } else {
                to.open = !ch.empty() && ch == popen[(size_t) from.open] ? from.open + 1 : again;
            }
        }
        bool quoted_tag = false;
        for (const auto& [i, j] : from.at) {
            const TextEnd& e = ends[(size_t) i];
            if (ch.empty()) continue;
            if (j < (int) e.chars.size() && e.chars[(size_t) j] == ch) {
                // a quoted parameter's tag: only the strings excluded after it too are followed further
                if (j + 1 == e.gap && from.depth > 0) quoted_tag = true;
                if (j + 1 != e.gap || from.depth == 0 || e.quoted) to.at.insert({i, j + 1});
            }
            if ((j == e.gap || j == e.gap2) && is_space(ch)) to.at.insert({i, j});   // white space before the structure
        }
        if (quoted_tag) to.depth = from.depth - 1;
        return to;
    };
    const auto complete = [&](const State& s) {
        for (const auto& [i, j] : s.at)
            if (j == (int) ends[(size_t) i].chars.size()) return true;
        return false;
    };
    const std::string quoted = add_quoted_call(c, form);
    std::string excluded;
    for (const std::string& a : alpha) excluded += a.size() == 1 ? hex_escape((unsigned char) a[0]) : a;
    std::map<State, int> ids;
    std::vector<State> states;
    const auto id_of = [&](const State& s) {
        const auto it = ids.find(s);
        if (it != ids.end()) return it->second;
        const int id = (int) states.size();
        ids.emplace(s, id);
        states.push_back(s);
        return id;
    };
    const auto rule = [&](const State& s) { return "free-text-" + std::to_string(id_of(s)); };
    State first;
    first.at = start();
    id_of(first);
    for (size_t at = 0; at < states.size(); ++at) {   // the states are found as their rules are written
        const State from = states[at];
        std::vector<std::string> alts;
        for (const std::string& a : alpha) {
            const State to = move(from, a);
            if (complete(to)) continue;
            alts.push_back(literal(a) + " " + rule(to));
        }
        alts.push_back("[^" + excluded + "] " + rule(move(from, "")));
        if (!quoted.empty()) {
            State after = first;
            after.depth = from.depth;
            alts.push_back(quoted + " " + rule(after));
        }
        alts.push_back("\"\"");
        c.add_rule("free-text-" + std::to_string(at), join(alts, " | "));
    }
    return "free-text-0";
}

// What closes a free-text value: where the text may hold the tag, the form's close from the tag on.  The text itself
// takes what the close writes before the tag (a line break), so the tag a model writes straight after a value - the
// one that closes a parameter the value quotes, when it leaves out its own - still closes it.
static std::string text_close(const ToolForm& form) {
    const size_t at = form.text_holds_end && !form.text_end.empty() ? form.param_close.find(form.text_end) : std::string::npos;
    return at == std::string::npos ? form.param_close : form.param_close.substr(at);
}

// The object variants of a oneOf/anyOf union, nested unions flattened.
void collect_variants(const JsonValue* root, const JsonValue* s, std::vector<const JsonValue*>& out, int depth) {
    s = deref(root, s);
    if (!s || !s->is(Type::Object) || depth > 8) return;
    bool is_union = false;
    for (const char* k : {"oneOf", "anyOf"}) {
        if (const JsonValue* u = s->get(k); u && u->is(Type::Array)) {
            is_union = true;
            for (const JsonValue& alt : u->items) collect_variants(root, &alt, out, depth + 1);
        }
    }
    const JsonValue* p = s->get("properties");
    if (!is_union && p && p->is(Type::Object)) out.push_back(s);
}

bool string_enum(const JsonValue& s) {
    const JsonValue* e = s.is(Type::Object) ? s.get("enum") : nullptr;
    return e && e->is(Type::Array) &&
           std::all_of(e->items.begin(), e->items.end(), [](const JsonValue& v) { return v.is(Type::String); });
}

JsonValue json_string(const std::string& text) {
    JsonValue v;
    v.type = Type::String;
    v.s = text;
    return v;
}

// Two schemas for one property name in different variants: string enums are united, anything else becomes an anyOf.
JsonValue merge_property(const JsonValue& cur, const JsonValue& add) {
    if (cur.dump() == add.dump()) return cur;
    if (string_enum(cur) && string_enum(add)) {
        JsonValue merged = cur;
        JsonValue en = *cur.get("enum");
        for (const JsonValue& e : add.get("enum")->items) {
            auto same = [&](const JsonValue& x) { return x.s == e.s; };
            if (std::none_of(en.items.begin(), en.items.end(), same)) en.items.push_back(e);
        }
        merged.set("enum", en);
        return merged;
    }
    JsonValue alts;
    alts.type = Type::Array;
    const JsonValue* any = cur.get("anyOf");
    if (any && any->is(Type::Array) && cur.members.size() == 1) alts = *any;
    else alts.items.push_back(cur);
    auto same = [&](const JsonValue& x) { return x.dump() == add.dump(); };
    if (std::none_of(alts.items.begin(), alts.items.end(), same)) alts.items.push_back(add);
    JsonValue merged;
    merged.type = Type::Object;
    merged.set("anyOf", alts);
    return merged;
}

// Parameters sent as a union of objects ({"properties": {}, "oneOf": [...]}, as zod-generated tools do) become one
// object: every property of every variant, and required only what every variant requires (the client still checks
// the variant). False when the parameters are not such a union.
bool flatten_union(const JsonValue* params, JsonValue& out) {
    if (!params || !params->is(Type::Object)) return false;
    if (const JsonValue* p = params->get("properties"); p && p->is(Type::Object) && !p->members.empty()) return false;
    std::vector<const JsonValue*> variants;
    for (const char* k : {"oneOf", "anyOf"}) {
        if (const JsonValue* u = params->get(k); u && u->is(Type::Array))
            for (const JsonValue& alt : u->items) collect_variants(params, &alt, variants, 0);
    }
    if (variants.empty()) return false;

    JsonValue props;
    props.type = Type::Object;
    std::map<std::string, size_t> required_by;   // property -> number of variants that require it
    for (const JsonValue* v : variants) {
        for (const auto& m : v->get("properties")->members) {
            const JsonValue* r = deref(params, &m.second);
            JsonValue value;
            if (r) value = *r;
            else value.type = Type::Object;
            const JsonValue* cur = props.get(m.first);
            props.set(m.first, cur ? merge_property(*cur, value) : value);
        }
        if (const JsonValue* req = v->get("required"); req && req->is(Type::Array)) {
            for (const JsonValue& n : req->items)
                if (n.is(Type::String)) ++required_by[n.s];
        }
    }

    JsonValue required;
    required.type = Type::Array;
    for (const auto& m : props.members) {
        auto it = required_by.find(m.first);
        if (it != required_by.end() && it->second == variants.size()) required.items.push_back(json_string(m.first));
    }
    out = JsonValue();
    out.type = Type::Object;
    out.set("type", json_string("object"));
    out.set("properties", props);
    out.set("required", required);
    for (const char* k : {"$defs", "definitions"})
        if (const JsonValue* d = params->get(k)) out.set(k, *d);
    return true;
}

}  // namespace

namespace {
ToolForm g_tool_form;
}

void set_tool_form(const ToolForm& form) { g_tool_form = form; }
const ToolForm& tool_form() { return g_tool_form; }

std::string tool_call_gbnf(const std::string& tools_json, const ToolGrammarOptions& opt, std::string& err) {
    const ToolForm& form = tool_form();
    if (form.call_open.empty()) {
        err = "tools: this model has no tool-call grammar";
        return "";
    }
    JsonValue tools;
    if (!read_json(tools_json, tools, err)) return "";
    if (!tools.is(Type::Array) || tools.items.empty()) {
        err = "tools: expected a non-empty array";
        return "";
    }
    SchemaConverter c;
    std::string text_rule;
    std::vector<std::string> fns;
    for (size_t ti = 0; ti < tools.items.size(); ++ti) {
        const JsonValue* tool = &tools.items[ti];
        if (const JsonValue* f = tool->is(Type::Object) ? tool->get("function") : nullptr) tool = f;
        const JsonValue* nm = tool->is(Type::Object) ? tool->get("name") : nullptr;
        if (!nm || !nm->is(Type::String) || nm->s.empty()) {
            err = "tools: every tool needs a name";
            return "";
        }
        const JsonValue* params = tool->get("parameters");
        if (!params) params = tool->get("input_schema");
        JsonValue flat;
        if (flatten_union(params, flat)) params = &flat;
        const JsonValue* props = params && params->is(Type::Object) ? params->get("properties") : nullptr;
        std::set<std::string> required;
        if (const JsonValue* r = params && params->is(Type::Object) ? params->get("required") : nullptr) {
            for (const JsonValue& n : r->items) if (n.is(Type::String)) required.insert(n.s);
        }
        std::string fn_id = "fn" + std::to_string(ti);
        if (form.json_args) {   // one JSON object of the parameter schema (any object when the tool declares none)
            JsonValue any;
            any.type = Type::Object;
            any.set("type", json_string("object"));
            const JsonValue* schema = params && params->is(Type::Object) ? params : &any;
            std::string args;
            try {
                SchemaDocument d = build_schema_document(*schema);
                args = c.add_schema(fn_id + "-args", *d.root);
            } catch (const std::exception& e) {
                err = "tools: " + nm->s + ": " + e.what();
                return "";
            }
            std::string mid = form.fn_mid;
            const std::string quoted = json_string(nm->s).dump();
            for (size_t p = mid.find("{name}"); p != std::string::npos; p = mid.find("{name}", p + quoted.size()))
                mid.replace(p, 6, quoted);
            fns.push_back(c.add_rule(fn_id, literal(form.fn_open + nm->s + mid) + " " + args + " " + literal(form.fn_close)));
            continue;
        }
        std::vector<std::string> req_items, opt_items;
        if (props && props->is(Type::Object)) {
            for (const auto& m : props->members) {
                const JsonValue* ps = deref(params, &m.second);
                JsonValue any_schema;   // `true` and a missing target are the empty schema
                if (ps && ps->is(Type::Bool) && !ps->b) {   // `false`: no value is valid
                    if (required.count(m.first)) {
                        err = "tools: " + nm->s + "." + m.first + ": required parameter has the schema false, no value is valid";
                        return "";
                    }
                    continue;
                }
                if (!ps || ps->is(Type::Bool)) {
                    any_schema.type = Type::Object;
                    ps = &any_schema;
                }
                const JsonValue* ty = ps && ps->is(Type::Object) ? ps->get("type") : nullptr;
                const JsonValue* en = ps && ps->is(Type::Object) ? ps->get("enum") : nullptr;
                bool raw = false;
                std::string value;
                bool typed = ps && ps->is(Type::Object) &&
                             (ps->get("anyOf") || ps->get("oneOf") || ps->get("allOf") || ps->get("const") ||
                              ps->get("properties") || ps->get("items"));
                if (en && en->is(Type::Array) && !en->items.empty()) {
                    bool all_str = std::all_of(en->items.begin(), en->items.end(),
                                               [](const JsonValue& v) { return v.is(Type::String); });
                    if (all_str) {
                        std::vector<std::string> vals;
                        for (const JsonValue& v : en->items) vals.push_back(literal(v.s));
                        value = c.add_rule(fn_id + "-" + m.first + "-enum", "(" + join(vals, " | ") + ")");
                        raw = true;
                    }
                } else if (ty && is_text_type(*ty)) {
                    raw = true;
                } else if (text_union(ps, params)) {
                    raw = true;
                } else if (!ty && !typed) {
                    raw = true;
                }
                bool free_text = false;
                if (raw && value.empty()) {
                    if (text_rule.empty()) text_rule = add_free_text(c, form);
                    value = text_rule;
                    free_text = true;
                }
                if (!raw) {
                    try {
                        SchemaDocument d = build_schema_document(*ps, params);
                        value = c.add_schema(fn_id + "-" + m.first, *d.root);
                    } catch (const std::exception& e) {
                        err = "tools: " + nm->s + "." + m.first + ": " + e.what();
                        return "";
                    }
                }
                if (value.empty()) {
                    err = "tools: " + nm->s + "." + m.first + ": no rule";
                    return "";
                }
                (required.count(m.first) ? req_items : opt_items)
                    .push_back(literal(form.param_open + m.first + (raw ? form.param_mid_text : form.param_mid_value)) + " " + value + " " +
                               literal(free_text ? text_close(form) : form.param_close));
            }
        }
        // As llama.cpp's Qwen3-Coder parser (common/chat-peg-parser.cpp permute, parsers/qwen3-coder.cpp): the required
        // parameters in any order, each once (one rule per set still to write; past max_permute they keep the declared
        // order), then the optional ones in any order, repeatable.
        std::string body;
        const size_t n = req_items.size();
        if (n == 1 || n > opt.max_permute) {
            for (const std::string& it : req_items) body += it + " ";
        } else if (n > 1) {
            std::vector<std::string> req_rule((size_t) 1 << n);
            for (size_t remaining = 1; remaining < ((size_t) 1 << n); ++remaining) {
                std::vector<std::string> alts;
                for (size_t i = 0; i < n; ++i) {
                    const size_t bit = (size_t) 1 << i;
                    if (!(remaining & bit)) continue;
                    alts.push_back(req_items[i] + (remaining & ~bit ? " " + req_rule[remaining & ~bit] : ""));
                }
                req_rule[remaining] = c.add_rule(fn_id + "-req" + std::to_string(remaining), join(alts, " | "));
            }
            body = req_rule[((size_t) 1 << n) - 1] + " ";
        }
        if (!opt_items.empty()) body += "(" + c.add_rule(fn_id + "-opt", join(opt_items, " | ")) + ")* ";
        fns.push_back(c.add_rule(fn_id, literal(form.fn_open + nm->s + form.fn_mid) + " " + body + literal(form.fn_close)));
    }
    const std::string one = "(" + join(fns, " | ") + ")";
    const bool shared_block = form.calls_share_block && opt.multiple_calls;
    const std::string call_rule =
        c.add_rule("call", literal(form.call_open) + " " + (shared_block ? one + " (" + literal("\n") + " " + one + ")*" : one) + " " + literal(form.call_close));
    std::string root = opt.multiple_calls && !form.calls_share_block ? call_rule + " ([ \\t\\n]* " + call_rule + ")*" : call_rule;
    if (!opt.answer_schema.empty()) {
        JsonValue answer;
        if (!read_json(opt.answer_schema, answer, err)) return "";
        try {
            SchemaDocument d = build_schema_document(answer);
            root = "(" + root + ") | " + c.add_schema("answer", *d.root);
        } catch (const std::exception& e) {
            err = std::string("schema: ") + e.what();
            return "";
        }
    }
    c.add_rule("root", root);
    try {
        c.check_errors();
    } catch (const std::exception& e) {
        err = std::string("tools: ") + e.what();
        return "";
    }
    return c.format_grammar();
}

RequestGrammar grammar_from_request(const std::string& grammar_json, std::string& err) {
    RequestGrammar out;
    JsonValue req;
    if (!read_json(grammar_json, req, err)) return out;
    const JsonValue* kind = req.is(Type::Object) ? req.get("kind") : nullptr;
    if (!kind || !kind->is(Type::String)) {
        err = "grammar: missing kind";
        return out;
    }
    if (const JsonValue* think = req.get("think"); think && think->is(Type::Bool)) out.think = think->b;
    std::string gbnf;
    if (kind->s == "json_object" || kind->s == "json_schema") {
        std::string schema = R"({"type":"object"})";
        if (kind->s == "json_schema") {
            const JsonValue* sch = req.get("schema");
            if (!sch) {
                err = "grammar: json_schema needs a schema";
                return out;
            }
            schema = sch->dump();
        }
        if (const JsonValue* tools = req.get("tools")) {   // tool calls or the JSON answer, from the first answer token
            if (!tools->is(Type::Array)) {
                err = "grammar: tools needs an array";
                return out;
            }
            ToolGrammarOptions opt;
            opt.answer_schema = schema;
            gbnf = tool_call_gbnf(tools->dump(), opt, err);
        } else {
            gbnf = json_schema_to_gbnf(schema, err);
        }
    } else if (kind->s == "tools") {
        const JsonValue* tools = req.get("tools");
        if (!tools || !tools->is(Type::Array)) {
            err = "grammar: tools needs an array";
            return out;
        }
        std::string only;   // a named choice
        if (const JsonValue* tc = req.get("tool_choice")) {
            if (tc->is(Type::String) && tc->s == "none") return out;
            if (tc->is(Type::Object)) {
                const JsonValue* fn = tc->get("function");
                const JsonValue* nm = fn && fn->is(Type::Object) ? fn->get("name") : tc->get("name");
                if (nm && nm->is(Type::String)) only = nm->s;
            }
        }
        JsonValue list = *tools;
        if (!only.empty()) {
            list.items.clear();
            for (const JsonValue& t : tools->items) {
                const JsonValue* f = t.is(Type::Object) && t.get("function") ? t.get("function") : &t;
                const JsonValue* nm = f->is(Type::Object) ? f->get("name") : nullptr;
                if (nm && nm->is(Type::String) && nm->s == only) list.items.push_back(t);
            }
        }
        gbnf = tool_call_gbnf(list.dump(), ToolGrammarOptions{}, err);
        if (!tool_form().trigger_pattern.empty()) out.triggers.patterns = {tool_form().trigger_pattern};
        else out.triggers.words = {tool_form().trigger};
    } else {
        err = "grammar: unknown kind '" + kind->s + "'";
        return out;
    }
    if (gbnf.empty()) return out;
    out.grammar = Grammar::parse(gbnf, err);
    return out;
}

}  // namespace basalt::program::grammar
