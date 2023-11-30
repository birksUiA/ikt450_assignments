#import "template.typ": *

#show: project.with(
  title: "Weekly Assignments",
  subtitle: "IKT450 - Deep Neural Networks",
  header: "IKT450 - Deep Neural Networks",
  authors: (
    (name: "Mathias Birk Olsen", email: "mathiasbol"),
  ),
)

#include "sections/0-abstract.typ"

#pagebreak()

#outline()
#pagebreak()

#include "sections/1-assignment.typ"
#pagebreak()
#include "sections/2-assignment.typ"
#pagebreak()
#include "sections/3-assignment.typ"
#pagebreak()
#include "sections/4-assignment.typ"
#pagebreak()
#include "sections/5-assignment.typ"
#pagebreak()
#include "sections/6-assignment.typ"
#pagebreak()
#include "sections/7-assignment.typ"
#pagebreak()
#bibliography("refrences.bib")
