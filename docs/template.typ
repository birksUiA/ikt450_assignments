#let project(title: "", authors: (), body) = {
  set page(
    margin: 1.25in,
    numbering: "1 of 1",
    header: align(right, "IKT450 Deep Neural Networks"),
  )
  
  set heading(numbering: "1.1 ")
  set par(
    leading: 0.55em,
    justify: true
  )
  set text(font: "Noto Sans", size: 10pt)
  
  show raw: set text(font: "JetBrainsMono Nerd Font")
  show raw.where(block: true): it => {
    set par(justify: false);
    set text(size: 8pt);
    
    block(radius: 1em, fill: luma(240), width: 100%, inset: 1em, it)
  }

  show par: set block(spacing: 1.5em)
  show heading: set block(above: 1.4em, below: 1em)

  align(center)[
    #pad(top: 2cm, bottom: 25pt, image("assets/uia-logo.png", width: 70%))
  ]

  // Title row
  align(center)[
    #pad(top: 0.5em, bottom: 2em)[
      #block(text(weight: 500, 1.75em, title))
      #pad(top: 1em, block(text(weight: 500, 1.2em, "Group Teapot")))
      #block(
        grid(
          columns: (1fr,) * authors.len(),
          gutter: 1.5em,
          ..authors.map(author => text(weight: 500, 1.2em, align(center)[
            *#author.name* \
            #author.email
          ]))
        )
      )
      #v(15pt)
      #block(text(weight: 500, 1.2em, "IKT450"))
      #block(text(weight: 500, 1.2em, datetime.today().display())) 
      #v(15pt)
    ]
  ]
  
  body
}
