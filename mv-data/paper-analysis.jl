using DataFrames, DataFramesMeta, CSV, Statistics

d = CSV.read("epri.csv")
hcb = CSV.read("hcb.csv")
ieeeall = CSV.read("ieeeall.csv")

using Gadfly
Gadfly.push_theme(Theme(grid_line_style = :solid))

function fig1()
    rng = [24, 0]
    case = (x = :IE, y = :IEmeas, color = :Iameas)
    plot(layer(filter(d-> d.config == "EPRI HCB", d), Geom.point; case...),
         layer(x = rng ./ 2  , y = rng, label = ["2x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("purple"))),
         layer(x = rng ./ 2.5, y = rng, label = ["2.5x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("darkblue"))),
         layer(x = rng ./ 3  , y = rng, label = ["3x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("blue"))),
         Guide.xlabel("Predicted, assuming HCB"), Guide.ylabel("Measured"),
         Guide.title("Incident energy, cal/cm²"),
         Guide.colorkey(title="Iₐ, kA"),
         Coord.Cartesian(xmin=0,xmax=12,ymin=0,ymax=25),
         style(plot_padding=[1mm,3mm,1mm,1mm]))
end

function fig2()
    rng = [24, 0]
    d.D_in = Int.(round.(d.D_mm ./ 25.4))
    case = (x = :IE, y = :IEmeas, color = :D_in)
    plot(layer(filter(d-> d.config == "EPRI HCB", d), Geom.point; case...),
         layer(x = rng ./ 2  , y = rng, label = ["2x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("purple"))),
         layer(x = rng ./ 2.5, y = rng, label = ["2.5x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("darkblue"))),
         layer(x = rng ./ 3  , y = rng, label = ["3x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("blue"))),
         Guide.xlabel("Predicted, assuming HCB"), Guide.ylabel("Measured"),
         Guide.title("Incident energy, cal/cm²"),
         Guide.colorkey(title="D, in"),
         Coord.Cartesian(xmin=0,xmax=12,ymin=0,ymax=25),
         style(plot_padding=[1mm,3mm,1mm,1mm]))
end

function fig3()
    rng = [10, 0]
    case = (x = :IE, y = :IEmeas, color = :Iameas)
    plot(layer(filter(d-> d.config == "EPRI VCB", d), Geom.point; case...),
         layer(x = rng, y = rng, Geom.line, Theme(default_color=color("purple"))),
         Guide.xlabel("Predicted, assuming VCB"), 
         Guide.ylabel("Measured"),
         Guide.title("Incident energy, cal/cm²"),
         Guide.colorkey(title="Iₐ, kA"),
         Coord.Cartesian(xmin=0,xmax=10,ymin=0,ymax=10),
         style(plot_padding=[1mm,3mm,1mm,1mm]))
end

function fig4()
    rng = [14, 0]
    case = (x = :IE, y = :IEmeas, color = :Iameas)
    plot(layer(filter(d-> d.config == "EPRI Transformer", d), Geom.point; case...),
         layer(x = rng, y = rng, label = ["1x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("purple"))),
         layer(x = rng ./ 1.5, y = rng, label = ["1.5x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("darkblue"))),
         Guide.xlabel("Predicted, assuming HCB"), 
         Guide.ylabel("Measured"),
         Guide.title("Incident energy, cal/cm²"),
         Guide.colorkey(title="Iₐ, kA"),
         style(plot_padding=[1mm,3mm,1mm,1mm]))
end

function fig5()
    rng = [19, 0]
    rng2 = [12, 0]
    case = (x = :IE, y = :IEmeas, color = :Iameas)
    plot(layer(dropmissing(filter(d-> d.config == "EPRI PMH-9", d), [:IE, :IEmeas, :Iameas]), Geom.point; case...),
         layer(x = rng2, y = rng2, label = ["1x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("purple"))),
         layer(x = rng ./ 2, y = rng, label = ["2x",""], Geom.line, Geom.label(position=:above), Theme(default_color=color("darkblue"))),
         Guide.xlabel("Predicted, assuming HCB"), 
         Guide.ylabel("Measured"),
         Guide.title("Incident energy, cal/cm²"),
         Guide.colorkey(title="Iₐ, kA"),
         Coord.Cartesian(xmin=0,xmax=12,ymin=0,ymax=20),
         style(plot_padding=[1mm,3mm,1mm,1mm]))
end

function fig6()
    dt = filter(d-> d.energyratio < 5, hcb)
    dt = @transform(dt, arcV = :joules ./ :t ./ :Iameas ./ 3)
    case = (x = :gap_mm, y = :arcV, color = :config, shape = :config)
    plot(dt,
        Geom.point,
        Guide.xlabel("Electrode gap, mm"), 
        Guide.ylabel("Calculated arc voltage, V"),
        style(point_shapes=[Shape.circle, Shape.square], alphas=[0.0], discrete_highlight_color=identity, highlight_width=(0.2mm), colorkey_swatch_shape=:circle, plot_padding=[1mm,3mm,1mm,1mm]);
        case...   
    )
end

function fig7()
    dt = filter(d-> d.energyratio < 5 && 35 < d.D_in < 40, hcb)
    case = (x = :t, y = :energyratio, color = :config, shape = :config)
    plot(dt,
        Geom.point, 
        Guide.xlabel("Duration, secs"), 
        Guide.ylabel("Energy ratio, cal/cm²/MJ"),
        Guide.title("Distance = 914 mm (36 in)"),
        style(point_shapes=[Shape.circle, Shape.square], alphas=[0.0], discrete_highlight_color=identity, highlight_width=(0.2mm), colorkey_swatch_shape=:circle, plot_padding=[1mm,3mm,1mm,1mm]);   
        case...   
    )
end

function fig8()
    dt = filter(d-> d.energyratio < 5 && 45 < d.D_in < 50, hcb)
    case = (x = :t, y = :energyratio, color = :config, shape = :config)
    plot(dt,
        Geom.point, 
        Guide.xlabel("Duration, secs"), 
        Guide.ylabel("Energy ratio, cal/cm²/MJ"),
        Guide.title("Distance = 1219 mm (48 in)"),
        style(point_shapes=[Shape.circle, Shape.square], alphas=[0.0], discrete_highlight_color=identity, highlight_width=(0.2mm), colorkey_swatch_shape=:circle, plot_padding=[1mm,3mm,1mm,1mm]);
        case...   
    )
end

function fig10()
    dt = filter(d-> d.energyratio < 5, hcb)
    dt = @transform(dt, It = :Iameas ./ :t, D_m = :D_in * 0.0254)
    case = (x = :D_m, y = :energyratio, color = :config, shape = :height_in)
    plot(dt,
        Geom.point, 
        Guide.xlabel("Distance, m"), 
        Guide.ylabel("Energy ratio, cal/cm²/MJ"),
        style(point_shapes=[Shape.circle, Shape.square], alphas=[0.0], discrete_highlight_color=identity, highlight_width=(0.2mm), colorkey_swatch_shape=:circle, plot_padding=[1mm,3mm,1mm,1mm]);
        case...   
    )
end

function fig12()
    dt = filter(d-> d.config == "VCB" && d.Voc > 1 && d.height_in == 36 && 18 < d.Ibf < 22, 
            dropmissing(ieeeall, [:height_in, :Voc, :config]))
    dt = @transform(dt, D_m = :D_mm / 1000, gap = CategoricalArray(:gap_mm), Voc = ifelse.(:Voc .> 10, "14 kV", "2.7 kV"))
    sizemap(p::Float64; min=0.5mm, max=2mm) = min + p*(max-min)
    case = (x = :D_m, y = :IErate, color = :Voc, size = :gap_mm)
    plot(dt,
        Scale.size_area(sizemap, minvalue=25, maxvalue=150),
        Geom.point, 
        Guide.sizekey(title="Gap, mm"),
        Guide.xlabel("Distance, m"), 
        Guide.ylabel("Heat rate, cal/cm²/sec"),
        style(highlight_width=(0.2mm), colorkey_swatch_shape=:circle, plot_padding=[1mm,3mm,1mm,1mm]);
        case...
    )
end

function fig13()
    dt = filter(d-> d.config == "HCB" && d.Voc > 1 && d.height_in == 36 && 18 < d.Ibf < 22, 
            dropmissing(ieeeall, [:height_in, :Voc, :config]))
    dt = @transform(dt, D_m = :D_mm / 1000, gap = CategoricalArray(:gap_mm), Voc = ifelse.(:Voc .> 10, "14 kV", "2.7 kV"))
    sizemap(p::Float64; min=0.5mm, max=2mm) = min + p*(max-min)
    case = (x = :D_m, y = :IErate, color = :Voc, size = :gap_mm)
    plot(dt,
        Scale.size_area(sizemap, minvalue=25, maxvalue=150),
        Geom.point, 
        Guide.sizekey(title="Gap, mm"),
        Guide.xlabel("Distance, m"), 
        Guide.ylabel("Heat rate, cal/cm²/sec"),
        style(highlight_width=(0.2mm), colorkey_swatch_shape=:circle, plot_padding=[1mm,3mm,1mm,1mm]);
        case...
    )
end


using GLM

function table1()
    dt = filter(d-> d.energyratio < 5 && d.config == "IEEE HCB", hcb)
    dt = @transform(dt, arcV = :joules ./ :t ./ :Iameas ./ 3)
    lm(@formula(log(arcV) ~ log(Iameas) + log(gap_mm)), dt)
end

function table2()
    dt = filter(d-> d.energyratio < 5 && d.config == "IEEE HCB", hcb)
    dt = @transform(dt, arcV = :joules ./ :t ./ :Iameas ./ 3)
    lm(@formula(log(IEmeas) ~ log(D_in) + log(Iameas) + log(t) + log(gap_mm)), dt)
end

function table3()
    dt = filter(d-> d.energyratio < 5 && d.config == "EPRI HCB", hcb)
    dt = @transform(dt, arcV = :joules ./ :t ./ :Iameas ./ 3)
    lm(@formula(log(IEmeas) ~ log(D_in) + log(Iameas) + log(t)), dt)
end

function table5()
    dt = filter(d-> d.config == "VCB" && d.Voc > 1 && d.height_in == 36, 
            dropmissing(ieeeall, [:height_in, :Voc, :config, :Iameas]));
    lm(@formula(log(IEmeas) ~ log(D_mm) + log(Ibf) + log(t) + log(gap_mm) + log(Voc)), dt)
end

function table6()
    dt = filter(d-> d.config == "HCB" && d.Voc > 1 && d.height_in == 36, 
            dropmissing(ieeeall, [:height_in, :Voc, :config, :Iameas]));
    lm(@formula(log(IEmeas) ~ log(D_mm) + log(Ibf) + log(t) + log(gap_mm) + log(Voc)), dt)
end

