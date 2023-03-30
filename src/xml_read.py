from lxml import etree
doc = etree.parse('EstimacionesTrafico.xml')
print(etree.tostring(doc,pretty_print=True ,xml_declaration=True, encoding="utf-8"))
raiz = doc.getroot()

for j in range(len(raiz)):
    dato2 = raiz[j]
    print(dato2.tag, j)
    for i in dato2:
        print('    ',i.tag,': ', i.text)