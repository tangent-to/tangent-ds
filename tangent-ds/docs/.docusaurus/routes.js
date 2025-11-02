import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/tangent-ds/blog',
    component: ComponentCreator('/tangent-ds/blog', '206'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/archive',
    component: ComponentCreator('/tangent-ds/blog/archive', '0de'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/authors',
    component: ComponentCreator('/tangent-ds/blog/authors', '28b'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/authors/all-sebastien-lorber-articles',
    component: ComponentCreator('/tangent-ds/blog/authors/all-sebastien-lorber-articles', '2b3'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/authors/yangshun',
    component: ComponentCreator('/tangent-ds/blog/authors/yangshun', 'c28'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/first-blog-post',
    component: ComponentCreator('/tangent-ds/blog/first-blog-post', 'ba7'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/long-blog-post',
    component: ComponentCreator('/tangent-ds/blog/long-blog-post', '9d2'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/mdx-blog-post',
    component: ComponentCreator('/tangent-ds/blog/mdx-blog-post', '604'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/tags',
    component: ComponentCreator('/tangent-ds/blog/tags', '72c'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/tags/docusaurus',
    component: ComponentCreator('/tangent-ds/blog/tags/docusaurus', 'f9e'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/tags/facebook',
    component: ComponentCreator('/tangent-ds/blog/tags/facebook', '8ab'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/tags/hello',
    component: ComponentCreator('/tangent-ds/blog/tags/hello', '7a7'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/tags/hola',
    component: ComponentCreator('/tangent-ds/blog/tags/hola', '86a'),
    exact: true
  },
  {
    path: '/tangent-ds/blog/welcome',
    component: ComponentCreator('/tangent-ds/blog/welcome', '8b6'),
    exact: true
  },
  {
    path: '/tangent-ds/markdown-page',
    component: ComponentCreator('/tangent-ds/markdown-page', '835'),
    exact: true
  },
  {
    path: '/tangent-ds/docs',
    component: ComponentCreator('/tangent-ds/docs', '6c4'),
    routes: [
      {
        path: '/tangent-ds/docs',
        component: ComponentCreator('/tangent-ds/docs', '6e4'),
        routes: [
          {
            path: '/tangent-ds/docs',
            component: ComponentCreator('/tangent-ds/docs', '02f'),
            routes: [
              {
                path: '/tangent-ds/docs/api/',
                component: ComponentCreator('/tangent-ds/docs/api/', '105'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/category/tutorial---basics',
                component: ComponentCreator('/tangent-ds/docs/category/tutorial---basics', 'c6a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/category/tutorial---extras',
                component: ComponentCreator('/tangent-ds/docs/category/tutorial---extras', 'd2b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/intro',
                component: ComponentCreator('/tangent-ds/docs/intro', '41e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/tutorial-basics/congratulations',
                component: ComponentCreator('/tangent-ds/docs/tutorial-basics/congratulations', 'ad1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/tutorial-basics/create-a-blog-post',
                component: ComponentCreator('/tangent-ds/docs/tutorial-basics/create-a-blog-post', '4df'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/tutorial-basics/create-a-document',
                component: ComponentCreator('/tangent-ds/docs/tutorial-basics/create-a-document', 'aa5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/tutorial-basics/create-a-page',
                component: ComponentCreator('/tangent-ds/docs/tutorial-basics/create-a-page', '635'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/tutorial-basics/deploy-your-site',
                component: ComponentCreator('/tangent-ds/docs/tutorial-basics/deploy-your-site', '78d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/tutorial-basics/markdown-features',
                component: ComponentCreator('/tangent-ds/docs/tutorial-basics/markdown-features', '24f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/tutorial-extras/manage-docs-versions',
                component: ComponentCreator('/tangent-ds/docs/tutorial-extras/manage-docs-versions', 'a90'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/tangent-ds/docs/tutorial-extras/translate-your-site',
                component: ComponentCreator('/tangent-ds/docs/tutorial-extras/translate-your-site', 'f7c'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/tangent-ds/',
    component: ComponentCreator('/tangent-ds/', '125'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
