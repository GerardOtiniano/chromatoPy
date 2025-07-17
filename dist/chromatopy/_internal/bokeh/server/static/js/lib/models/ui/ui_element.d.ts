import { Model } from "../../model";
import { Node } from "../coordinates/node";
import { Styles } from "../dom/styles";
import type { Menu } from "./menus/menu";
import { StyleSheet as BaseStyleSheet } from "../dom/stylesheets";
import type { Align } from "../../core/enums";
import type { SizingPolicy } from "../../core/layout";
import type { ViewOf } from "../../core/view";
import { DOMComponentView } from "../../core/dom_view";
import type { SerializableState } from "../../core/view";
import type { StyleSheet, StyleSheetLike } from "../../core/dom";
import { InlineStyleSheet } from "../../core/dom";
import { CanvasLayer } from "../../core/util/canvas";
import type { XY } from "../../core/util/bbox";
import { BBox } from "../../core/util/bbox";
import type * as p from "../../core/properties";
export declare const StylesLike: import("../../core/kinds").Kinds.Or<[import("../../core/types").Dict<string | null>, Styles]>;
export type StylesLike = typeof StylesLike["__type__"];
export declare const StyleSheets: import("../../core/kinds").Kinds.List<string | BaseStyleSheet | import("../../core/types").Dict<Styles | import("../../core/types").Dict<string | null>>>;
export type StyleSheets = typeof StyleSheets["__type__"];
export declare const CSSVariables: import("../../core/kinds").Kinds.Dict<Node>;
export type CSSVariables = typeof CSSVariables["__type__"];
export type DOMBoxSizing = {
    width_policy: SizingPolicy | "auto";
    height_policy: SizingPolicy | "auto";
    width: number | null;
    height: number | null;
    aspect_ratio: number | "auto" | null;
    halign?: Align;
    valign?: Align;
};
export declare abstract class UIElementView extends DOMComponentView {
    model: UIElement;
    protected readonly _display: InlineStyleSheet;
    readonly style: InlineStyleSheet;
    protected _css_classes(): Iterable<string>;
    protected _css_variables(): Iterable<[string, string]>;
    protected _stylesheets(): Iterable<StyleSheet>;
    protected _computed_stylesheets(): Iterable<StyleSheet>;
    stylesheets(): StyleSheetLike[];
    update_style(): void;
    box_sizing(): DOMBoxSizing;
    private _bbox;
    get bbox(): BBox;
    update_bbox(): boolean;
    protected _update_bbox(): boolean;
    protected _resize_observer: ResizeObserver;
    protected _context_menu: ViewOf<Menu> | null;
    initialize(): void;
    lazy_initialize(): Promise<void>;
    connect_signals(): void;
    get_context_menu(_xy: XY): ViewOf<Menu> | null;
    show_context_menu(event: MouseEvent): void;
    remove(): void;
    protected _after_resize(): void;
    after_resize(): void;
    render(): void;
    protected _after_render(): void;
    after_render(): void;
    private _is_displayed;
    get is_displayed(): boolean;
    protected _apply_visible(): void;
    protected _apply_styles(): void;
    protected _update_visible(): void;
    protected _update_styles(): void;
    export(type?: "auto" | "png" | "svg", hidpi?: boolean): CanvasLayer;
    serializable_state(): SerializableState;
    resolve_symbol(node: Node): XY | number;
}
export declare namespace UIElement {
    type Attrs = p.AttrsOf<Props>;
    type Props = Model.Props & {
        visible: p.Property<boolean>;
        css_classes: p.Property<string[]>;
        css_variables: p.Property<CSSVariables>;
        styles: p.Property<StylesLike>;
        stylesheets: p.Property<StyleSheets>;
        context_menu: p.Property<Menu | null>;
    };
}
export interface UIElement extends UIElement.Attrs {
}
export declare abstract class UIElement extends Model {
    properties: UIElement.Props;
    __view_type__: UIElementView;
    constructor(attrs?: Partial<UIElement.Attrs>);
}
//# sourceMappingURL=ui_element.d.ts.map